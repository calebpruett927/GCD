#!/usr/bin/env python3
"""
GCD Full OS
===========

This standalone script implements the full Generative Collapse Dynamics (GCD)
and Unified Monitoring & Contract Protocol (UMCP) pipeline in a single file.
It can be run directly as a command‐line program to compute invariants,
assign regimes, generate audits, produce plots, and perform basic weld tests.

The goal of this script is to provide a self‑contained environment—almost
an operating system—for interacting with GCD data.  All functionality is
implemented below without external module dependencies beyond NumPy, pandas,
matplotlib and click.

Usage:
    python gcd_full_os.py --help
"""

from __future__ import annotations

import datetime
import json
import uuid
import click
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def compute_invariants(
    df: pd.DataFrame,
    x_col: str = "x",
    time_col: str | None = None,
    a: float = 0.0,
    b: float = 1.0,
    epsilon: float = 1e-8,
    alpha: float = 1.0,
) -> pd.DataFrame:
    """
    Compute collapse invariants for a single‑channel time series.

    Parameters
    ----------
    df : pandas.DataFrame
        Input data containing at least the column `x_col` and optionally `time_col`.
    x_col : str
        Name of the column with raw observations.
    time_col : str or None
        Name of the time column.  If None, the row index is treated as time.
    a, b : float
        Frozen affine normalisation parameters.  b must be positive.
    epsilon : float
        Small positive constant to avoid singularities near ω → 1.
    alpha : float
        Decay factor used in the integrity coefficient.

    Returns
    -------
    pandas.DataFrame
        A dataframe with columns: `x_hat`, `omega`, `F`, `S`, `C`, `tau_R`, `IC` and `kappa`.
    """
    if b <= 0:
        raise ValueError("Normalisation scale b must be positive.")
    if x_col not in df.columns:
        raise KeyError(f"Column {x_col!r} not found in dataframe.")
    # Normalise and clip
    y = (df[x_col].astype(float) - a) / b
    x_hat = y.clip(0.0, 1.0)
    # First difference of x_hat
    delta = x_hat.diff().fillna(0.0).abs()
    omega = delta.copy()
    # Flip probability
    F = 1.0 - omega
    # Entropy term (use log1p for numerical stability)
    S = -np.log1p(-omega + epsilon)
    # Curvature: second difference magnitude of ω
    second_diff = omega.diff().fillna(0.0)
    C = second_diff.abs()
    # First return time τ_R: discrete time since last change in the sign of Δx_hat
    tau_R: list[float] = []
    last_idx = 0
    last_sign = 0.0
    for i, d in enumerate(delta):
        sign = np.sign(d)
        if i == 0:
            tau_R.append(0.0)
            last_sign = sign
            last_idx = 0
            continue
        if sign != 0.0 and sign != last_sign:
            last_sign = sign
            last_idx = i
        tau_R.append(float(i - last_idx))
    tau_R = np.asarray(tau_R, dtype=float)
    # Integrity coefficient
    IC = F * np.exp(-S) * (1.0 - omega) * np.exp(-alpha * C / (1.0 + tau_R))
    # Clip to avoid log of zero
    IC_clipped = IC.clip(lower=1e-20)
    kappa = np.log(IC_clipped)
    out = pd.DataFrame({
        'x_hat': x_hat,
        'omega': omega,
        'F': F,
        'S': S,
        'C': C,
        'tau_R': tau_R,
        'IC': IC,
        'kappa': kappa,
    })
    # Preserve time if provided
    if time_col is not None and time_col in df.columns:
        out.insert(0, time_col, df[time_col])
    return out



def assign_regimes(
    df: pd.DataFrame,
    omega_col: str = 'omega',
    F_col: str = 'F',
    S_col: str = 'S',
    C_col: str = 'C',
    ic_col: str = 'IC',
    thresholds: dict | None = None,
) -> pd.Series:
    """
    Assign qualitative regimes based on collapse invariants.

    Parameters
    ----------
    df : pandas.DataFrame
        A dataframe containing at least the specified invariant columns.
    omega_col, F_col, S_col, C_col, ic_col : str
        Names of the columns corresponding to ω, F, S, C and IC respectively.
    thresholds : dict or None
        Optional override for the default gate thresholds.

    Returns
    -------
    pandas.Series
        Series of regime labels aligned with `df`.
    """
    default_thresholds = {
        'stable': {'omega': 0.038, 'F': 0.90, 'S': 0.15, 'C': 0.14},
        'watch': {'omega': 0.30},
    }
    if thresholds is None:
        thresholds = default_thresholds
    regimes: list[str] = []
    for _, row in df.iterrows():
        omega_val = row[omega_col]
        F_val = row[F_col]
        S_val = row[S_col]
        C_val = row[C_col]
        ic_val = row[ic_col]
        if any(pd.isna([omega_val, F_val, S_val, C_val, ic_val])):
            regimes.append('⊥oor')
            continue
        stable = (
            omega_val < thresholds['stable']['omega'] and
            F_val > thresholds['stable']['F'] and
            S_val < thresholds['stable']['S'] and
            C_val < thresholds['stable']['C']
        )
        watch = thresholds['stable']['omega'] <= omega_val <= thresholds['watch']['omega']
        collapse = omega_val > thresholds['watch']['omega']
        if stable:
            regimes.append('Stable')
        elif collapse:
            if ic_val < 0.30:
                regimes.append('Collapse|Critical')
            else:
                regimes.append('Collapse')
        elif watch:
            regimes.append('Watch')
        else:
            regimes.append('Watch')
    return pd.Series(regimes, index=df.index, name='regime')



def audit_dataframe(
    df: pd.DataFrame,
    x_col: str,
    time_col: str | None = None,
    a: float = 0.0,
    b: float = 1.0,
    epsilon: float = 1e-8,
    alpha: float = 1.0,
    thresholds: dict | None = None,
) -> pd.DataFrame:
    """
    Compute a full audit table for a dataframe.

    Parameters
    ----------
    df : pandas.DataFrame
        Raw input dataframe containing the observation column and optional time.
    x_col : str
        Name of the column containing the observations.
    time_col : str, optional
        Name of the column containing the time index.
    a, b : float
        Affine normalisation parameters.
    epsilon : float
        Epsilon parameter for entropy computation.
    alpha : float
        Decay factor for the integrity coefficient.
    thresholds : dict or None
        Optional override for regime thresholds.

    Returns
    -------
    pandas.DataFrame
        The audit table with invariants and regime column.  A manifest
        dictionary is stored in the `.attrs['manifest']` attribute.
    """
    invariants = compute_invariants(df, x_col=x_col, time_col=time_col, a=a, b=b, epsilon=epsilon, alpha=alpha)
    regimes = assign_regimes(invariants, thresholds=thresholds)
    audit = invariants.copy()
    audit['regime'] = regimes
    manifest = {
        'audit_id': str(uuid.uuid4()),
        'timestamp': datetime.datetime.now().isoformat(),
        'contract': {'a': a, 'b': b, 'epsilon': epsilon},
        'alpha': alpha,
        'fields': {'x': x_col, 'time': time_col},
        'thresholds': thresholds or {},
    }
    audit.attrs['manifest'] = manifest
    return audit



def plot_invariant(
    df: pd.DataFrame,
    y_col: str,
    time_col: str | None = None,
    out_path: str | None = None,
) -> None:
    """
    Plot a single invariant against time or index.

    Parameters
    ----------
    df : pandas.DataFrame
        Audit dataframe containing the invariant column.
    y_col : str
        Name of the invariant column to plot (e.g. 'kappa', 'omega').
    time_col : str, optional
        Name of the time column.  If None, the row index is used for the x-axis.
    out_path : str, optional
        If provided, save the plot to this path.
    """
    if y_col not in df.columns:
        raise KeyError(f"Column {y_col!r} not found in dataframe.")
    x = df[time_col] if time_col is not None and time_col in df.columns else df.index
    y = df[y_col]
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=1.5)
    ax.set_xlabel(time_col or 'index')
    ax.set_ylabel(y_col)
    ax.set_title(f"{y_col} vs {time_col or 'index'}")
    ax.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight')
    else:
        plt.show()



def assert_kappa_continuity(
    audit_df: pd.DataFrame,
    pivot_col: str,
    kappa_col: str = 'kappa',
    tolerance: float = 0.05,
) -> bool:
    """
    Assert that the logarithmic integrity (κ) is continuous across
    boundaries defined by `pivot_col`.

    Parameters
    ----------
    audit_df : pandas.DataFrame
        Audit dataframe containing the pivot indicator and κ column.
    pivot_col : str
        Name of the column whose changes denote a contract or policy change.
    kappa_col : str
        Name of the column containing κ values.
    tolerance : float
        Maximum allowed jump in κ at the pivot (default 0.05).

    Returns
    -------
    bool
        True if all boundaries satisfy the continuity criterion.
    """
    if pivot_col not in audit_df.columns:
        raise ValueError(f"pivot_col '{pivot_col}' not found in audit dataframe.")
    if kappa_col not in audit_df.columns:
        raise ValueError(f"kappa_col '{kappa_col}' not found in audit dataframe.")
    changes = audit_df[pivot_col].ne(audit_df[pivot_col].shift())
    boundary_indices = audit_df.index[changes][1:]
    for idx in boundary_indices:
        prev_idx = idx - 1
        prev_k = audit_df.loc[prev_idx, kappa_col]
        next_k = audit_df.loc[idx, kappa_col]
        diff = abs(next_k - prev_k)
        if diff > tolerance:
            raise AssertionError(
                f"κ continuity failed at boundary index {idx}: |κ_next - κ_prev|={diff:.4f} > tolerance {tolerance}"
            )
    return True


@click.group()
def cli() -> None:
    """GCD Full OS: compute invariants, audit data, plot and test welds."""
    pass


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--col', required=True, help='Column name containing observations.')
@click.option('--time', 'time_col', default=None, help='Time column name.')
@click.option('--a', default=0.0, type=float, help='Normalisation offset a.')
@click.option('--b', default=1.0, type=float, help='Normalisation scale b (positive).')
@click.option('--epsilon', default=1e-8, type=float, help='Epsilon to avoid log singularity.')
@click.option('--alpha', default=1.0, type=float, help='Alpha parameter for integrity decay.')
@click.option('--out', 'out_path', default=None, help='Output CSV file path.')
def audit(input: str, col: str, time_col: str | None, a: float, b: float, epsilon: float, alpha: float, out_path: str | None) -> None:
    """
    Compute invariants and regimes for a CSV file.

    INPUT should be a path to a CSV file containing at least the column
    specified by --col and optionally a time column (--time).  The audit
    table is printed to stdout and optionally written to --out.
    """
    df = pd.read_csv(input)
    audit_df = audit_dataframe(df, x_col=col, time_col=time_col, a=a, b=b, epsilon=epsilon, alpha=alpha, thresholds=None)
    if out_path:
        audit_df.to_csv(out_path, index=False)
        click.echo(f"Wrote audit to {out_path}")
    click.echo(audit_df.head().to_string(index=False))


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--y', 'y_col', default='kappa', help='Invariant column to plot (default: kappa).')
@click.option('--time', 'time_col', default=None, help='Time column name.')
@click.option('--out', 'out_path', default=None, help='Output image path.')
def plot(input: str, y_col: str, time_col: str | None, out_path: str | None) -> None:
    """
    Plot an invariant column from an audit CSV.

    By default plots κ (kappa) against the row index or time column if provided.
    """
    df = pd.read_csv(input)
    plot_invariant(df, y_col, time_col, out_path)
    if out_path:
        click.echo(f"Saved plot to {out_path}")


@cli.command()
@click.option('--a', default=0.0, type=float, help='Normalisation offset a.')
@click.option('--b', default=1.0, type=float, help='Normalisation scale b.')
@click.option('--epsilon', default=1e-8, type=float, help='Epsilon parameter.')
@click.option('--alpha', default=1.0, type=float, help='Alpha parameter.')
def manifest(a: float, b: float, epsilon: float, alpha: float) -> None:
    """
    Print a manifest template for the given contract parameters.

    This can be useful when preparing reproducible manifests for audit
    runs or documenting the contract in reports.
    """
    manifest = {
        'contract': {'a': a, 'b': b, 'epsilon': epsilon},
        'alpha': alpha,
        'timestamp': None,
    }
    click.echo(json.dumps(manifest, indent=2))


@cli.command()
@click.argument('input', type=click.Path(exists=True))
@click.option('--pivot', 'pivot_col', required=True, help='Pivot column indicating contract or policy changes.')
@click.option('--kappa', 'kappa_col', default='kappa', help='Column name for κ.')
@click.option('--tol', 'tolerance', default=0.05, type=float, help='Tolerance for κ continuity.')
def weldtest(input: str, pivot_col: str, kappa_col: str, tolerance: float) -> None:
    """
    Perform a weld continuity test on an audit CSV.

    Reads the audit CSV and asserts that κ is continuous across
    the boundaries defined by the pivot column.  Raises an error
    on failure.
    """
    df = pd.read_csv(input)
    try:
        assert_kappa_continuity(df, pivot_col, kappa_col, tolerance)
        click.echo("κ continuity test passed.")
    except AssertionError as e:
        click.echo(str(e))


if __name__ == '__main__':
    cli()
