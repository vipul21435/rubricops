#!/bin/sh
# End-to-end demo on the bundled examples: validate and diff two versions of a
# rubric, score one review against it, measure inter-rater agreement with
# bootstrap intervals, assign a review queue, list overdue work and sample
# reviews for QA, calibrate reviewers against gold items and feed the flags into
# the QA sampler, then run the review pipeline on a fresh SQLite database and
# verify its audit chain. Runs from a checkout (make demo) or inside the image
# (make docker-demo), where RUBRICOPS=rubricops and the examples sit in the cwd.
set -eu

RUBRICOPS=${RUBRICOPS:-uv run rubricops}
EX=${EXAMPLES:-examples}

step() {
    printf '\n$ rubricops %s\n' "$*"
    # shellcheck disable=SC2086 # RUBRICOPS may be a multi-word command
    $RUBRICOPS "$@"
}

step --version
step rubric validate "$EX/rubrics/action-items.yaml" "$EX/rubrics/code-explanation.v1.yaml" \
    "$EX/rubrics/code-explanation.yaml"
step rubric diff "$EX/rubrics/code-explanation.v1.yaml" "$EX/rubrics/code-explanation.yaml"
step rubric score "$EX/rubrics/code-explanation.yaml" \
    --scores "$EX/reviews/code-explanation-review.yaml"
step agreement "$EX/ratings/correctness-3-reviewers.csv" \
    --metric alpha-interval --metric alpha-nominal
step agreement "$EX/ratings/verdicts-long.csv" --layout long \
    --metric cohen --metric alpha-nominal
step queue assign "$EX/queue/scenario.yaml" --policy skill-match
step queue overdue "$EX/queue/scenario.yaml"
step queue sample "$EX/queue/scenario.yaml" --seed 20260929 --rate 0.1

# Files the demo writes go to a throwaway directory outside the (possibly read-only) cwd.
DB_DIR=$(mktemp -d)
trap 'rm -rf "$DB_DIR"' EXIT

CAL="--gold $EX/calibration/gold.yaml --reviews $EX/calibration/reviews.yaml"
CAL="$CAL --rubric $EX/rubrics/code-explanation.yaml"
# shellcheck disable=SC2086 # CAL is a list of options
step calibration report $CAL --reviewer chen
# shellcheck disable=SC2086
$RUBRICOPS calibration report $CAL --format json >"$DB_DIR/calibration.json"
step queue sample "$EX/queue/scenario.yaml" --seed 20260929 --rate 0.1 \
    --flags "$DB_DIR/calibration.json"

DB_URL="sqlite:///$DB_DIR/walkthrough.db"
step pipeline walkthrough --url "$DB_URL" \
    --rubric "$EX/rubrics/code-explanation.yaml" --scores "$EX/reviews/walkthrough-scores.yaml"
step audit verify --url "$DB_URL"
printf '\ndemo finished\n'
