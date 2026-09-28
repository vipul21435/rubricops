#!/bin/sh
# End-to-end demo on the bundled examples: validate and diff two versions of a
# rubric, score one review against it, then measure inter-rater agreement with
# bootstrap intervals. Runs from a checkout (make demo) or inside the image
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
step agreement "$EX/ratings/gwet-abstractors.csv" \
    --metric ac1 --metric cohen
step agreement "$EX/ratings/correctness-3-reviewers.csv" \
    --metric ac2-quadratic --metric alpha-interval
printf '\ndemo finished\n'
