# syntax=docker/dockerfile:1
# Slim runtime image for the rubricops CLI.
#   docker build -t rubricops:dev .
#   docker run --rm rubricops:dev agreement examples/ratings/correctness-3-reviewers.csv
#   docker run --rm -v "$PWD:/work" -w /work rubricops:dev rubric validate my-rubric.yaml
# Both base images are pinned by digest so a rebuild gets the same bits.

FROM ghcr.io/astral-sh/uv:0.11.29@sha256:eb2843a1e56fd9e30c7276ce1a52cba86e64c7b385f5e3279a0e08e02dd058fc AS uv

FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f AS builder
LABEL project=rubricops
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src
# Dependencies first so source edits do not invalidate this layer.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

FROM python:3.12-slim@sha256:f77ac9e44ae96ef2c90b8053ea08c31f8be030f824196b0ae4db6d462c84e51f
LABEL project=rubricops \
      org.opencontainers.image.title="rubricops" \
      org.opencontainers.image.description="Versioned rubrics, rubric scoring and inter-rater agreement for expert review operations" \
      org.opencontainers.image.source="https://github.com/vipul21435/rubricops" \
      org.opencontainers.image.licenses="MIT"
RUN useradd --create-home --uid 10001 --shell /usr/sbin/nologin rubricops
COPY --from=builder /opt/venv /opt/venv
COPY examples /opt/rubricops/examples
COPY scripts/demo.sh /opt/rubricops/scripts/demo.sh
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1
USER rubricops
# Relative example paths in the usage above resolve from here.
WORKDIR /opt/rubricops
ENTRYPOINT ["rubricops"]
CMD ["--help"]
