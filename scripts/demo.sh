#!/bin/sh
# End-to-end demo on the bundled examples: validate and diff two versions of a
# rubric, score one review against it, measure inter-rater agreement with
# bootstrap intervals, then run the review pipeline on a fresh SQLite database and
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

# The pipeline writes to a throwaway database outside the (possibly read-only) cwd.
DB_DIR=$(mktemp -d)
trap 'rm -rf "$DB_DIR"' EXIT
DB_URL="sqlite:///$DB_DIR/walkthrough.db"
step pipeline walkthrough --url "$DB_URL" \
    --rubric "$EX/rubrics/code-explanation.yaml" --scores "$EX/reviews/walkthrough-scores.yaml"
step audit verify --url "$DB_URL"
printf '\ndemo finished\n'
