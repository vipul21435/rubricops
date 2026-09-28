"""Builders for rubric documents used across the test suite.

They return plain dicts (the shape a YAML file parses into), so tests can break one
field at a time and check that validation rejects it.
"""

from __future__ import annotations

from typing import Any

from rubricops.domain.rubric import Rubric


def criterion_dict(
    cid: str = "accuracy",
    *,
    weight: float = 1.0,
    low: int = 1,
    high: int = 4,
    gating: int | None = None,
    title: str | None = None,
) -> dict[str, Any]:
    """A valid criterion with one descriptor and one anchor per scale point."""
    data: dict[str, Any] = {
        "id": cid,
        "title": title or cid.replace("_", " ").capitalize(),
        "weight": weight,
        "scale": {"low": low, "high": high},
        "guide": [
            {"score": s, "label": f"level {s}", "descriptor": f"{cid} at level {s}"}
            for s in range(low, high + 1)
        ],
        "anchors": [
            {"score": s, "text": f"example of {cid} at {s}", "rationale": f"why {s}"}
            for s in range(low, high + 1)
        ],
    }
    if gating is not None:
        data["gating"] = gating
    return data


def rubric_dict(
    *criteria: dict[str, Any],
    rid: str = "sample-rubric",
    threshold: float = 0.6,
) -> dict[str, Any]:
    """A valid rubric; with no criteria given it has two, weighted 0.6 and 0.4."""
    if not criteria:
        criteria = (
            criterion_dict("accuracy", weight=0.6, gating=2),
            criterion_dict("clarity", weight=0.4, low=0, high=2),
        )
    return {
        "id": rid,
        "title": "Sample rubric",
        "description": "A rubric used by the test suite.",
        "pass_threshold": threshold,
        "criteria": list(criteria),
    }


def make_rubric(*criteria: dict[str, Any], threshold: float = 0.6) -> Rubric:
    return Rubric.model_validate(rubric_dict(*criteria, threshold=threshold))
