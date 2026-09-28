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

> **Status:** under active development; features land slice by slice as described in
> [PLAN.md](PLAN.md). Merged so far: the scaffold (packaging, strict typing, lint,
> tests, CI) and **versioned rubrics with the scoring engine** (slice 1). This README
> only claims what is merged.

## Rubrics: validate, version, diff and score

A rubric is a YAML (or JSON) document. Each criterion has a weight, an integer scale,
a scoring guide with one descriptor per scale point, at least one anchor example per
point, and an optional `gating` minimum:

```yaml
id: code-explanation
title: Explaining a code snippet to a learner
pass_threshold: 0.7
criteria:
  - id: correctness
    title: Technical correctness
    weight: 0.4            # weights are fractions of the total and must sum to 1
    scale: {low: 1, high: 4}
    gating: 3              # below 3 fails the review, whatever the total
    guide:
      - {score: 1, label: Wrong, descriptor: The core behaviour is described incorrectly.}
      # ... one entry per scale point
    anchors:
      - score: 1
        text: Says chunk() splits the list into `size` equal parts.
        rationale: It produces parts of length `size`, not `size` parts.
      # ... at least one per scale point
```

Full examples live in [examples/rubrics/](examples/rubrics/): a code-explanation
rubric (current version plus its v1, to show a diff) and an action-item extraction
rubric. All three are original.

**Validation** (pydantic, `rubricops.domain.rubric`) rejects duplicate criterion ids,
weights that do not sum to 1, guides that miss, repeat or overshoot a scale point,
anchors outside the scale, scale points with no anchor, gating minimums off the
scale or at its bottom (a gate that can never fail), a `pass_threshold` outside
[0, 1], unknown keys, and duplicate keys in the YAML or JSON file itself.

**Versioning** (`rubricops.domain.versioning`). The content hash is the sha256 of a
canonical JSON form (sorted keys, no insignificant whitespace, guide levels in scale
order), so reordering keys or re-indenting a file never creates a new version.
`RubricRegistry` keeps an append-only history: each publish is version n+1 with a
changelog message and timestamp, re-publishing an unchanged head is refused, and a
revert becomes a new version that shares the old hash.

**Scoring** (`rubricops.domain.scoring`). A review is a `criterion -> int` map. Each
score is normalised to `(s - low) / (high - low)`, and the weighted mean lies in
[0, 1]. A review passes when that mean reaches `pass_threshold` and no gate fails.
Weights and thresholds are combined as exact decimals rather than binary floats, so
a score that equals the threshold on paper passes. Hypothesis property tests check
that the score stays in [0, 1] and rises strictly with every criterion.

`make demo` runs these commands; the output shown is real:

```console
$ uv run rubricops rubric validate examples/rubrics/*.yaml
ok    examples/rubrics/action-items.yaml  id=action-items criteria=4 pass_threshold=0.75 sha256=14e0f1332a8a
ok    examples/rubrics/code-explanation.v1.yaml  id=code-explanation criteria=3 pass_threshold=0.6 sha256=772003b4c66e
ok    examples/rubrics/code-explanation.yaml  id=code-explanation criteria=4 pass_threshold=0.7 sha256=15f1cde6e5fe

$ uv run rubricops rubric diff examples/rubrics/code-explanation.v1.yaml examples/rubrics/code-explanation.yaml
code-explanation (772003b4c66e) -> code-explanation (15f1cde6e5fe)
  + added criterion safe_advice: weight 0.15, scale 0..2, gating 1
  ~ reweighted correctness: 0.5 -> 0.4
  ~ reweighted completeness: 0.3 -> 0.25
  ~ rescaled clarity: 1..3 -> 1..4
  ~ gating correctness: none -> 3
  ~ pass_threshold: 0.6 -> 0.7
  ~ guide or anchors edited: completeness
scoring affected: yes

$ uv run rubricops rubric score examples/rubrics/code-explanation.yaml \
    --scores examples/reviews/code-explanation-review.yaml
rubric code-explanation  sha256 15f1cde6e5fe
criterion     score  scale   weight  contribution
correctness       3  1..4     0.400        0.2667
completeness      3  1..4     0.250        0.1667
clarity           4  1..4     0.200        0.2000
safe_advice       1  0..2     0.150        0.0750
score 0.7083  threshold 0.7  -> PASS
```

Other options:
- `rubricops rubric score RUBRIC -s correctness=2 ...` scores inline, and overrides a
  `--scores` file.
- `--json` on `diff` and `score` prints machine-readable output.
- `rubricops rubric diff --fail-on-scoring-change OLD NEW` exits 1 when an edit can
  change a score or a verdict. It is meant as a CI gate: rewording a descriptor passes,
  reweighting does not.

Exit codes: 0 success, 1 invalid rubric or scores (every problem is listed, not just
the first), 2 usage error.

## Quickstart

Requires [uv](https://docs.astral.sh/uv/) and GNU make.

```bash
git clone https://github.com/vipul21435/rubricops.git
cd rubricops
make install   # uv sync --frozen + pre-commit hooks
make check     # ruff, mypy --strict, pytest with coverage gate
make demo      # validate, diff and score the example rubrics
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
src/rubricops/
  domain/        pure logic: rubric models, versioning, diff, scoring, clock
  cli/           Typer commands (rubricops rubric validate|diff|score)
  loaders.py     strict YAML/JSON loading (duplicate keys rejected)
  settings.py    typed RUBRICOPS_* configuration
examples/        original example rubrics and a sample review
tests/           pytest + hypothesis suite
PLAN.md          design decisions and the implementation slices
```

## License

MIT. See [LICENSE](LICENSE).
