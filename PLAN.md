# RubricOps implementation plan

RubricOps is the operations layer a review squad lead runs to keep AI-training data
quality high across many expert reviewers: versioned rubrics, a review queue, a
primary review -> QA audit -> adjudication pipeline, reviewer calibration and
inter-rater agreement. Every rubric, dataset and example in this repo is original.

## Decisions

- **Fresh repo, no fork.** A GitHub search for a base turned up only single-file
  agreement-statistic scripts (mostly GPL) and nothing platform-shaped. Building on
  them would not save work and would pull in an incompatible license, so the repo
  starts fresh under MIT.
- **Layering.** Pure, DB-free domain code does the reasoning; services compose it
  with persistence; the API and UI stay thin.
  - `rubricops.domain`: rubrics, scoring, pipeline state machine, queue policies,
    calibration
  - `rubricops.stats`: agreement statistics and bootstrap
  - `rubricops.db`: SQLAlchemy 2 models, session, repositories
  - `rubricops.services`: transactions that combine domain and db
  - `rubricops.api` and `rubricops.web`: FastAPI routers, Jinja2 + HTMX templates
- **Dependencies stay light.** Add them only in the slice that needs them:
  - pyyaml (slice 1)
  - numpy for vectorised statistics (slice 2); no scipy, pandas or ML libraries
  - sqlalchemy and alembic (slice 3)
  - fastapi, uvicorn, jinja2, pyjwt and python-multipart (slice 6)
  - psycopg[binary], only as the optional `postgres` extra (slice 7)

  Passwords are hashed with the stdlib `hashlib.scrypt`, so there is no passlib or
  bcrypt. Dev-only extras are hypothesis and httpx.
- **Determinism.**
  - Every RNG (QA sampling, bootstrap, the demo seed) is a seeded
    `numpy.random.Generator` or `random.Random`, built from `RUBRICOPS_RANDOM_SEED`
    or an explicit argument.
  - Time comes from an injected `Clock` protocol (a `SystemClock` in production and
    a `FrozenClock` in tests).
  - All datetimes are timezone-aware UTC; ruff's DTZ rule enforces this.
- **Scores.**
  - Each criterion is scored on an integer scale declared by the rubric, for example
    1..5.
  - The weighted score is normalised to [0, 1]. A review passes when the score is at
    or above the rubric's `pass_threshold` and every criterion marked `gating`
    reaches its own minimum.
- **Rubric versions are immutable.** Each version stores the canonical JSON body and
  its sha256 content hash. An edit creates version n+1. Reviews, gold items and
  submissions reference `rubric_version_id`, never the mutable rubric head.
- **The audit log is append-only and tamper-evident.**
  - Each event stores `prev_hash` and `hash`, where `hash = sha256(prev_hash +
    canonical event)`.
  - The ORM refuses UPDATE and DELETE, and so do database triggers created in the
    migration (SQLite and Postgres).
  - `verify_audit_chain()` reports the first broken link.
- **Roles.**
  - `author` submits items.
  - `reviewer` does primary reviews and QA audits, never on their own work.
  - `lead` adjudicates, manages rubrics and sees the dashboard.
- **Terminology.** Unusual reviewers or scores are called "anomalous" or "extreme"
  in code and docs, to stay clear of a reserved term.
- **Quality bar for every slice.**
  - ruff, mypy --strict and pytest with the 85% branch-coverage gate stay green on
    every commit.
  - Each slice begins by fixing any bug the previous slice's reviewer confirmed,
    with a regression test.
  - README claims only what is merged, and every number in it comes from a command
    shown next to it.

## Slices

### Slice 1: Versioned rubrics and scoring engine [x] done

Decisions made while building it:
- Weights are fractions of the total and must sum to 1 as written (tolerance 1e-9).
  Each float is read back as the shortest decimal that round-trips it, so
  `0.7 + 0.2 + 0.1` sums to exactly 1 even though it does not in binary floating
  point.
- Scoring uses the same exact decimals and `fractions.Fraction`, so a score equal
  to the threshold on paper passes. Floats appear only in reported values.
- A scale has at most 11 points (0..10), because every point needs a descriptor and
  an anchor. A `gating` minimum at the bottom of the scale is rejected, since it
  could never fail.
- Anchors live in a per-criterion list with an explicit `score`, rather than nested
  under guide levels, so "anchor outside the scale" is a real, checkable error.
- Canonical form: text is trimmed, guide levels are sorted by score, and anchors are
  stably sorted by score. Criterion order is kept, because it is the order a review
  form shows, so reordering criteria is a new version and the diff reports it.
- Publishing a document identical to the head raises `UnchangedRubricError`. A revert
  is a new version that shares an older hash (`find_by_hash` returns both).
- The diff reports a rescaled criterion only under `rescaled`, not also as a guide
  edit. `affects_scoring` is true for added or removed criteria, weight, scale,
  gating and threshold changes, and false for wording, anchor and order changes.
- The YAML and JSON loaders reject duplicate keys (YAML merge-key overrides are still
  allowed). CLI exit codes are 0 for success, 1 for invalid input or a gated diff,
  and 2 for usage errors.
- The `Clock` protocol (`SystemClock`, `FrozenClock`) landed here because the
  registry timestamps versions.

Goal: model rubrics as validated, immutable, versioned documents and score reviews
against them.
- `rubricops.domain.rubric` defines pydantic models `Rubric`, `Criterion`,
  `ScaleLevel` and `AnchorExample`:
  - a criterion has an id, a title, a weight > 0, an integer scale, an optional
    `gating` minimum, and a scoring guide with one descriptor per scale point and at
    least one anchor example per level
  - the rubric has a `pass_threshold` in [0, 1]
- Validators reject:
  - duplicate criterion ids
  - weights that do not normalise
  - guides that miss a scale point
  - anchors outside the scale
- Canonical JSON and a sha256 content hash make versions stable when keys are
  reordered.
- A `RubricRegistry` keeps append-only version history with a changelog message. A
  structural diff between versions reports added and removed criteria, reweighting,
  scale changes and threshold changes.
- The scoring engine validates a criterion -> int score map against a specific
  version and returns a `ScoreResult`: the normalised weighted score, each
  criterion's contribution, gating failures and pass/fail.
- CLI: `rubricops rubric validate|diff|score`, plus two original example rubrics
  under `examples/rubrics/` (YAML).

Commits:
- (a) models and validation
- (b) hashing, registry and diff
- (c) scoring engine
- (d) CLI and examples

Tests:
- every validation error path
- hash stability
- diff cases
- hypothesis properties: the score stays in [0, 1] and is monotone in each criterion
  score

### Slice 2: Agreement statistics with bootstrap confidence intervals [x] done

Decisions made while building it:
- Ratings are encoded once (`ReliabilityData`) against a fixed, ordered category
  list, so a bootstrap resample keeps the category set, and therefore the weights,
  of the full data.
- Weighted kappa weights by rating *value* over the declared span, so unused scale
  points change nothing and quadratic-weighted kappa equals Lin's concordance.
- Fleiss' kappa requires the same number of ratings on every rated unit and raises
  with a pointer to Krippendorff's alpha otherwise, instead of silently dropping
  units.
- The bootstrap requires a seed. A resample with no defined statistic is skipped and
  counted in `n_degenerate`; an undefined point estimate skips resampling. "The CI
  contains the estimate" is not true in general for percentile intervals, so it is
  exposed as `CIResult.contains_estimate` and tested on moderate data rather than
  claimed as a universal property.
- The ratings CSV loader reads a wide (unit then one column per rater) or long
  (unit, rater, rating) layout and, like the YAML loader, rejects rather than
  guesses: repeated units or (unit, rater) pairs, ragged rows and malformed quoting
  are errors with line numbers.
- The CLI accepts all six registered metrics (`cohen`, `cohen-linear`,
  `cohen-quadratic`, `fleiss`, `alpha-nominal`, `alpha-interval`), repeatable. A
  metric that cannot apply to the data is an error on its own line (exit 1) while the
  others still print.

Goal: `rubricops.stats.agreement` implements, in numpy:
- Cohen's kappa for two raters: unweighted, plus linear and quadratic weighted
- Fleiss' kappa for a fixed number of raters per item
- Krippendorff's alpha, nominal and interval, via the coincidence matrix. It handles
  missing ratings and varying raters per unit, and drops units with fewer than two
  ratings.

`rubricops.stats.bootstrap` resamples units (items) with a seeded generator and
returns a `CIResult` with the point estimate, percentile interval, confidence level,
number of resamples and number of degenerate resamples skipped.

Degenerate inputs have defined, documented behaviour: perfect agreement, a single
category or zero expected disagreement returns NaN with a reason rather than raising.

CLI: `rubricops agreement FILE.csv --metric cohen|fleiss|alpha-nominal|alpha-interval --ci 0.95`.

Commits:
- (a) Cohen
- (b) Fleiss
- (c) Krippendorff
- (d) bootstrap and CLI

Tests:
- Published worked examples, with the source cited in each test docstring. This
  includes Krippendorff's 2011 reliability-data example: nominal 0.743, interval
  0.849.
- Hand-computed small cases.
- Properties: invariance under relabelling for nominal metrics, kappa = 1 for
  identical raters, and the CI contains the point estimate.

### Slice 3: Persistence and the review pipeline state machine with an append-only audit log [x] done

Decisions made while building it:
- The state machine was built first (pure, no DB), so the models could reuse its
  `Role`, `Status` and `Stage` enums for their CHECK constraints.
- Actions: `assign`, `release`, `start`, `submit_primary`, `return_to_author`,
  `send_to_qa`, `finalize`, `submit_qa`, `escalate`, `adjudicate`, `resubmit`. All
  110 (status, action) pairs and the role matrix are tested against a table written
  out independently in the test.
- Independence guards beyond the two in the goal: nobody reviews, audits or
  adjudicates their own submission; only the assignee starts, submits or returns a
  primary review (a lead cannot submit someone else's); the adjudicator also must not
  be the QA auditor. Returning an item from adjudication is lead-only.
- QA passes only when the verdicts match *and* the scores are within
  `qa_tolerance` (default 0.1), compared as exact decimals so 0.8 vs 0.7 is exactly
  0.1 apart. Same score but a failed gate is a disagreement.
- `resubmit` starts a new `round`; reviews and assignments carry the round, so an
  earlier draft's reviews never count, and (submission, round, stage) is unique.
- The audit log is one global chain. `seq` is assigned by the writer (head + 1) so it
  is part of the hashed content, and UNIQUE on `prev_hash` and `hash` makes a fork
  impossible to commit. `verify_audit_chain` re-reads rows with `populate_existing`
  so it checks what is stored, not what the session cached. Truncating the newest
  events leaves a valid shorter chain, so the report prints an anchor (`seq:hash`)
  that can be passed back with `--anchor` to catch it.
- `rubric_versions` is append-only too (ORM guards, bulk-statement guard and
  triggers), and the service re-checks a version's body against its hash before
  scoring with it.
- Optimistic locking: `apply()` takes the caller's `expected_version` and checks it
  first; SQLAlchemy's `version_id_col` catches a writer that races between the read
  and the write. Both surface as `StaleSubmission`.
- Migrations ship inside the package (no `alembic.ini` needed) and run via
  `rubricops db upgrade|downgrade|current`. The migration inlines its trigger SQL;
  a test checks that a migrated database and a `create_all` database have identical
  triggers, and another that autogenerate finds zero drift.
- Postgres trigger SQL exists in the migration and models but is not exercised yet;
  the Postgres test job is part of slice 7.
- A deterministic `rubricops pipeline walkthrough` (a `SteppingClock` from a fixed
  instant) was added so the README and `make demo` show real pipeline output.


Goal: `rubricops.db` provides an engine and session factory from settings (SQLite
pragmas: foreign_keys=ON and WAL; any SQLAlchemy URL) and typed SQLAlchemy 2
`Mapped[]` models:
- `User`: role and skill tags
- `Rubric` and `RubricVersion`: immutable body and hash, unique (rubric_id, version)
- `Submission`: author, rubric_version_id, payload, skill tags, status, version
  column
- `Assignment`
- `Review`: stage primary|qa|adjudication, rubric_version_id, criterion scores,
  normalised score, pass
- `GoldItem`
- `AuditEvent`

There are Alembic migrations wired to the settings URL.

`rubricops.domain.pipeline` is the explicit status state machine:
- statuses: queued -> assigned -> in_review -> reviewed -> qa_pending ->
  qa_passed | qa_failed -> in_adjudication -> finalized, plus returned_to_author
- a transition table maps (status, action) to the next status, the allowed roles and
  guards:
  - the QA auditor is not the primary reviewer
  - the adjudicator is a lead
  - adjudication is required when the QA score differs by more than the tolerance
- illegal transitions raise `IllegalTransition` or `Forbidden`

`rubricops.services.pipeline` applies transitions in one DB transaction:
- it writes Review rows pinned to the submission's rubric version and a hash-chained
  `AuditEvent` for every transition
- it resolves the final score: the primary score if QA passes, the adjudicated score
  otherwise
- optimistic locking on the submission version column rejects a double submit

Commits:
- (a) engine, session and models
- (b) Alembic initial migration and migration tests (upgrade, downgrade, upgrade,
  plus a schema-drift check against the metadata)
- (c) the pure state machine with an exhaustive transition-table test
- (d) the pipeline service with the append-only hash-chained audit log (ORM guards
  plus DB triggers, `verify_audit_chain`, a tamper test) and hypothesis random-walk
  properties: nothing reaches finalized without a primary review, and there is one
  audit event per applied transition

### Slice 4: Review queue: assignment policies, SLAs and QA sampling [x] done

Decisions made while building it:
- The capacity cap applies to every policy, not only `LoadBalanced`: a reviewer at
  their cap takes no more work whichever policy is in use. Exclusions are checked in
  a fixed order (author, primary reviewer, at capacity; then missing skills for
  `SkillTagMatch`) and the first that applies is reported.
- The round-robin cursor is the id of the last reviewer assigned, not a list index,
  so a reviewer joining or leaving does not shift everyone's turn. It needs no table
  of its own: the CLI prints it for the caller to keep, and the queue service reads
  it from the latest primary assignment row.
- `assign_batch` counts each assignment towards the reviewer's open load for the
  rest of the batch, so caps and balancing hold across a run.
- Round-robin ignores skill tags by design; `SkillTagMatch` is the policy for
  tagged work.
- SLA: an assignment due exactly now is not late. The escalation list is sorted by
  lateness, then submission id, then reviewer id, so it is a total order.
- QA sampling draws from `random.Random(f"{seed}:{submission_id}:{round}")`, so a
  decision depends only on the seed and the item, never on batch order or size,
  and a resubmission gets a fresh draw. The draw is taken even when a risk rule
  fires, so the random part stays an unbiased sample, and it is recorded.
- "Low agreement between reviewers on the item" is measured as the spread (max -
  min) of the normalised scores recorded on the item; it fires above
  `max_score_spread` (default 0.25). The near-threshold rule compares exact
  decimals, so a score exactly `margin` away counts as near.
- The calibration hook is a `flagged` boolean per reviewer until slice 5 computes it.
- The queue CLI runs on a scenario file (YAML or JSON, strict like the rubric
  loader) so policies can be compared without a database.
- `rubricops.services.queue.QueueService` acts only through `PipelineService.apply`,
  which gained a `context` argument stored on the audit event: `assign_next` records
  the policy and cursor, `sample_for_qa` the reasons, draw, rate and seed, then
  sends the item to QA or finalizes it. `PipelineService` also takes an `SlaPolicy`,
  so `due_at` honours per-rubric overrides. `assign_next` handles primary
  assignments; a QA audit is still taken by any eligible reviewer under the
  pipeline's guards. For low agreement the service uses every other review score on
  the item, including earlier rounds.
- Still open: the queue CLI reads scenario files; a `--url` mode that runs the same
  commands against the database.

Goal: `rubricops.domain.queue` defines an `AssignmentPolicy` protocol with three
policies:
- `RoundRobin`: a stable cursor, persisted
- `SkillTagMatch`: reviewer tags must be a superset of the submission tags;
  tie-break by open load, then id
- `LoadBalanced`: fewest open assignments, with per-reviewer capacity caps

All policies exclude the author and, for QA, the primary reviewer, and return a
typed `NoEligibleReviewer` when nobody fits.

SLAs:
- `due_at` comes from settings or a per-rubric override
- overdue detection uses the injected Clock
- an escalation list is sorted by lateness

QA sampling combines a seeded random rate (`RUBRICOPS_QA_SAMPLE_RATE`) with
risk-based rules:
- a reviewer with fewer than N completed reviews
- a reviewer flagged by calibration (a hook, filled in slice 5)
- a normalised score within a margin of the pass threshold
- low agreement between reviewers on the item

Every sampling decision records its reasons.

`rubricops.services.queue` provides `assign_next`, `sweep_overdue` and
`sample_for_qa`, wired into the pipeline. CLI: `rubricops queue assign|overdue|sample`.

Commits:
- (a) assignment policies
- (b) SLA and overdue detection
- (c) the QA sampler with reasons
- (d) service wiring and CLI

Tests:
- policy fairness and eligibility
- FrozenClock SLA cases
- the sampling rate falls within binomial bounds for a fixed seed
- the same seed gives the same decisions

### Slice 5: Reviewer calibration and scorecards

Goal: `rubricops.domain.calibration` computes, from gold items with known
per-criterion scores:
- **Accuracy per reviewer:** exact-match rate, per-criterion MAE and pass/fail
  agreement.
- **Per-criterion bias:** the mean signed error with a bootstrap CI. It is
  classified lenient or harsh only when the CI excludes 0, and neutral otherwise.
- **Drift over time:** recent versus previous windows of gold reviews, a
  difference-in-means test with a configurable threshold, emitting `DriftAlert`s.
- **Reviewer scorecards:**
  - mean quality score, from QA and adjudication outcomes
  - review count and throughput
  - on-time rate against the SLA
  - gold accuracy
  - agreement with peers: pairwise Cohen's kappa and Krippendorff's alpha on
    overlapping items, from slice 2

The calibration flags feed the QA risk sampler from slice 4.

`rubricops.services.calibration` persists snapshots so drift is computed from
history. CLI: `rubricops calibration report [--reviewer ID] [--format table|json]`.

Commits:
- (a) gold accuracy
- (b) bias with CIs
- (c) drift detection
- (d) scorecards and feedback into the sampler

Tests: synthetic reviewers with a known injected leniency or harshness are detected,
unbiased reviewers are not flagged, and an injected step change triggers drift.

### Slice 6: FastAPI service with JWT roles, HTMX reviewer queue and lead dashboard, and export

Goal: `rubricops.api` has an app factory with dependency-injected settings,
sessions and clock.

Auth:
- JWT (HS256, PyJWT) with exp/iat/sub/role claims
- scrypt password hashing
- role dependencies for author, reviewer and lead
- 401 or 403 semantics, tested

REST routers:
- rubrics: create a version, list history, diff
- submissions
- queue: next assignment, overdue
- reviews: primary, QA, adjudication, through the pipeline service
- calibration and scorecards
- health

Every router validates with pydantic schemas and maps domain errors to 409 or 422.

`rubricops.web` renders server-side with Jinja2 + HTMX (vendored htmx.min.js, no JS
build):
- a reviewer queue page
- a review form rendered from the rubric version, with anchor examples inline
- a lead dashboard with throughput per day, quality (pass rate, mean score, agreement
  with a CI) and backlog (by status, overdue count)
- all of it behind cookie auth, with CSRF protection on forms

CSV and JSONL export of reviews, scorecards and the audit log, streamed through both
the API and `rubricops export`.

Commits:
- (a) app factory, auth and users
- (b) domain routers
- (c) HTMX UI and dashboard
- (d) export

Tests: httpx TestClient on SQLite, a role matrix test, the UI rendering with the
expected HTMX attributes, and an export round trip.

### Slice 7: Seeded demo dataset, Docker/compose and an end-to-end make demo

Pulled forward (done): a CLI-only image (digest-pinned python:3.12-slim and uv, uv
sync --frozen, non-root, `LABEL project=rubricops`), `scripts/demo.sh` behind
`make demo` and `make docker-demo`, and a CI job that builds the image and runs the
demo in it. Still open: everything below that needs the service (compose, Postgres,
HEALTHCHECK, the seeded HTTP demo).

Goal: `rubricops seed` generates an original, deterministic demo idempotently:
- rubrics
- authors, and reviewers with latent leniency, harshness and noise profiles
- submissions
- gold items

Docker:
- a multi-stage slim Dockerfile with a digest-pinned Python base
- uv sync --frozen, a non-root user, a HEALTHCHECK and `LABEL project=rubricops`
- docker-compose.yml runs the app plus a digest-pinned Postgres, and applies Alembic
  migrations on start
- the `postgres` extra adds psycopg

`make demo`:
- migrates, seeds, starts the app and drives the full pipeline over HTTP (assign,
  review, QA sample, audit, adjudicate)
- prints the dashboard numbers and exports
- checks that calibration recovers the seeded biased reviewers

`make demo-docker` does the same against compose and Postgres.

CI:
- a job that builds the image and runs a smoke test
- a job that runs the `postgres`-marked tests against a service container

Commits:
- (a) seed generator
- (b) Dockerfile and compose
- (c) demo script and make targets
- (d) CI jobs

Tests: an in-process e2e demo on SQLite, and seed idempotency. Clean up images with
`docker image prune -f --filter label=project=rubricops`.

### Slice 8: Benchmarks and documentation polish

Goal: `benchmarks/` holds reproducible scripts, and `make bench` writes
`benchmarks/results.md`:
- agreement-statistic runtime across item and rater counts
- bootstrap resamples per second
- assignment and QA-sampling throughput for 10k queued items
- in-process API p50/p95 latency for the queue and review endpoints

`docs/` holds:
- `architecture.md`: an ASCII layer diagram and the request flow
- `pipeline.md`: a state-machine diagram and transition table, generated from code
  and checked by a test so it cannot drift
- `statistics.md`: formulas, degenerate-case behaviour and the references used by
  the tests
- ADRs for the key decisions above

The README gets its final feature list with real numbers: test count, coverage, and
benchmark numbers with the commands that produce them. It also gets a screenshot-free
walkthrough of the UI and a fresh-clone check that the 5-command quickstart works.

Commits:
- (a) benchmarks and `make bench`
- (b) the docs set and the generated pipeline doc test
- (c) the README final pass and a CHANGELOG
