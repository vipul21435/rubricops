# RubricOps

[![CI](https://github.com/vipul21435/rubricops/actions/workflows/ci.yml/badge.svg)](https://github.com/vipul21435/rubricops/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

**Expert review and rubric-grading operations for AI-training data.**

When dozens of expert reviewers grade model outputs against rubrics, data quality is
decided by the process around the grading: which rubric version a score was given
under, who reviewed what, which items get audited, how disagreements are settled, and
which reviewers are drifting. RubricOps is that process as software. It is the
workflow a review squad lead runs every day, made explicit, typed and tested.

## What it does

| Area | What you get |
| --- | --- |
| Versioned rubrics | Weighted criteria, scoring guides with anchor examples, pass thresholds; pydantic-validated, content-hashed, with version history so every review stays pinned to the rubric version it used |
| Review queue | Assignment policies (round-robin, skill-tag match, load-balanced), SLAs with overdue detection, QA sampling (random rate plus risk-based: new reviewers, low-agreement items) |
| Review pipeline | Primary review -> QA audit -> adjudication as an explicit state machine; illegal transitions are rejected; every transition lands in an append-only audit log |
| Calibration | Gold items with known scores, reviewer accuracy against gold, per-criterion leniency/harshness bias, drift over time, reviewer scorecards |
| Agreement statistics | Cohen's kappa, Fleiss' kappa, Krippendorff's alpha (nominal and interval) with seeded bootstrap confidence intervals |
| Service | FastAPI + SQLAlchemy 2 (SQLite by default, Postgres by URL), Alembic migrations, JWT auth with author/reviewer/lead roles, a server-rendered Jinja2 + HTMX reviewer queue and lead dashboard, CSV/JSONL export, a seeded demo dataset |

> **Status:** under active development. The scaffold (packaging, strict typing, lint,
> tests, CI) is in place; features land slice by slice as described in
> [PLAN.md](PLAN.md). This README only claims what is merged.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and GNU make.

```bash
git clone https://github.com/vipul21435/rubricops.git
cd rubricops
make install   # uv sync --frozen + pre-commit hooks
make check     # ruff, mypy --strict, pytest with coverage gate
make demo      # end-to-end demo
```

## Development

```bash
make help       # list targets
make format     # apply ruff fixes and formatting
make typecheck  # mypy --strict on src/
make cov        # tests with branch coverage (fails under 85%)
```

Configuration is read from `RUBRICOPS_*` environment variables or a `.env` file; see
[.env.example](.env.example). Defaults are safe for local use, and `RUBRICOPS_ENV=prod`
refuses to start with the development JWT secret.

## Project layout

```
src/rubricops/   application package (typed, mypy --strict)
tests/           pytest suite
PLAN.md          design decisions and the implementation slices
```

## License

MIT. See [LICENSE](LICENSE).
