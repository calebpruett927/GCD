# GCD • UMCP • RCFT • ULRC — Starter Repository

A **production-grade**, auditable Python toolkit and repo scaffold for **Generative Collapse Dynamics (GCD)**,
including the **Universal Mathematics & Collapse Platform (UMCP)** invariants, **Recursive Collapse Field Theory (RCFT)**,
and **ULRC** governance hooks. Batteries included: CLI, tests, docs, examples, Docker, CI, and a reproducible
audit pipeline.

> Axiom: *collapse is generative; only that which returns through collapse is real.*

## Quickstart

```bash
# 1) Create env (any workflow works: UV, conda, venv). Example with uv:
uv venv
source .venv/bin/activate
uv pip install -e .

# 2) Run unit tests
pytest -q

# 3) Audit a CSV (see examples/data/sine_switch.csv)
umcp audit examples/data/sine_switch.csv --col x --time t --out artifacts/audit.csv

# 4) Plot invariants and regimes
umcp plot artifacts/audit.csv --out artifacts/audit.png
```

## What’s inside

- **src/umcp/** — Contract-first invariants (ω, F, S, C, τ_R, IC, κ), regime assignment, weld tests, plotting, and CLI.
- **examples/** — Synthetic datasets and runnable recipes.
- **artifacts/** — Output folder (ignored by git) for audits, plots, manifests.
- **tests/** — Pytest suite covering core invariants and regime logic.
- **docs/** — Lightweight docs (MkDocs-ready).
- **.github/workflows/** — CI using Python 3.11+ (lint + tests).
- **Dockerfile** — Reproducible container.
- **pyproject.toml** — Build, runtime deps, entry points.
- **Makefile** — Common tasks.

## CLI overview

```bash
umcp audit <csv> --col <name> --time <name> --out audit.csv [--a <val> --b <val> --mode global_fixed|p1p99]
umcp plot <audit.csv> --out audit.png
umcp manifest <audit.csv> --out manifest.json
```

- **audit**: normalizes, computes invariants (ω,F,S,C,τ_R,IC,κ), assigns regimes, writes a tidy CSV.
- **plot**: quick matplotlib chart (signal + IC + regime bands) for reports.
- **manifest**: records the frozen contract, defaults, hashes, and meta for reproducibility.

## Regime gates (defaults)

- Stable: ω < 0.038, F > 0.90, S < 0.15, C < 0.14
- Watch: 0.038 ≤ ω < 0.30
- Collapse: ω ≥ 0.30
- Critical when IC < 0.30

Tune via `--gates-config docs/gates.yaml` or edit `umcp/regimes.py` to publish deviations.

## License

MIT — see [LICENSE](LICENSE).

---

*Built to be hacked on. Add your closures, weld protocols, or domain adapters without breaking invariant math.*
