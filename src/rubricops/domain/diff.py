"""Structural diff between two rubric documents.

The diff answers the question a squad lead asks before publishing a new rubric
version: "are scores given under the old version still comparable with scores
under the new one?" Changes to weights, scales, gating, the pass threshold or the
set of criteria change the numbers (:attr:`RubricDiff.affects_scoring`). Edits to
titles, descriptors and anchor examples only change the guidance reviewers read.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from rubricops.domain.rubric import Criterion, Rubric, Scale


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CriterionRef(_Frozen):
    """The scoring-relevant facts about an added or removed criterion."""

    id: str
    title: str
    weight: float
    scale: Scale
    gating: int | None

    @classmethod
    def of(cls, criterion: Criterion) -> CriterionRef:
        return cls(
            id=criterion.id,
            title=criterion.title,
            weight=criterion.weight,
            scale=criterion.scale,
            gating=criterion.gating,
        )

    def describe(self) -> str:
        gating = f", gating {self.gating}" if self.gating is not None else ""
        return f"{self.id}: weight {self.weight:g}, scale {self.scale}{gating}"


class WeightChange(_Frozen):
    criterion_id: str
    old: float
    new: float


class ScaleChange(_Frozen):
    criterion_id: str
    old: Scale
    new: Scale


class GatingChange(_Frozen):
    criterion_id: str
    old: int | None
    new: int | None


class ThresholdChange(_Frozen):
    old: float
    new: float


class RubricDiff(_Frozen):
    """Everything that differs between an old and a new rubric document.

    ``guide_changed`` lists criteria whose descriptors or anchors were edited while
    the scale stayed the same; a rescaled criterion always has a new guide, so it is
    reported once, under ``rescaled``.
    """

    added: tuple[CriterionRef, ...] = ()
    removed: tuple[CriterionRef, ...] = ()
    reweighted: tuple[WeightChange, ...] = ()
    rescaled: tuple[ScaleChange, ...] = ()
    gating_changed: tuple[GatingChange, ...] = ()
    threshold: ThresholdChange | None = None
    guide_changed: tuple[str, ...] = ()
    retitled: tuple[str, ...] = ()
    reordered: bool = False
    metadata_changed: tuple[str, ...] = ()

    @property
    def is_empty(self) -> bool:
        """True when the two documents are identical in content."""
        return self == RubricDiff()

    @property
    def affects_scoring(self) -> bool:
        """True when the same review could get a different score or verdict."""
        return bool(
            self.added
            or self.removed
            or self.reweighted
            or self.rescaled
            or self.gating_changed
            or self.threshold is not None
        )

    def summary_lines(self) -> list[str]:
        """Human-readable change list: ``+`` added, ``-`` removed, ``~`` changed."""
        if self.is_empty:
            return ["no changes"]
        lines = [f"+ added criterion {ref.describe()}" for ref in self.added]
        lines += [f"- removed criterion {ref.describe()}" for ref in self.removed]
        lines += [f"~ reweighted {c.criterion_id}: {c.old:g} -> {c.new:g}" for c in self.reweighted]
        lines += [f"~ rescaled {c.criterion_id}: {c.old} -> {c.new}" for c in self.rescaled]
        lines += [
            f"~ gating {c.criterion_id}: {_gate(c.old)} -> {_gate(c.new)}"
            for c in self.gating_changed
        ]
        if self.threshold is not None:
            lines.append(f"~ pass_threshold: {self.threshold.old:g} -> {self.threshold.new:g}")
        if self.guide_changed:
            lines.append(f"~ guide or anchors edited: {', '.join(self.guide_changed)}")
        if self.retitled:
            lines.append(f"~ retitled: {', '.join(self.retitled)}")
        if self.reordered:
            lines.append("~ criteria reordered")
        if self.metadata_changed:
            lines.append(f"~ rubric metadata changed: {', '.join(self.metadata_changed)}")
        return lines


def _gate(value: int | None) -> str:
    return "none" if value is None else str(value)


def diff_rubrics(old: Rubric, new: Rubric) -> RubricDiff:
    """Compare two rubric documents criterion by criterion (matched on id)."""
    old_by_id = {c.id: c for c in old.criteria}
    new_by_id = {c.id: c for c in new.criteria}
    common = [cid for cid in old.criterion_ids if cid in new_by_id]

    reweighted: list[WeightChange] = []
    rescaled: list[ScaleChange] = []
    gating: list[GatingChange] = []
    guide: list[str] = []
    retitled: list[str] = []
    for cid in common:
        before, after = old_by_id[cid], new_by_id[cid]
        if before.weight != after.weight:
            reweighted.append(WeightChange(criterion_id=cid, old=before.weight, new=after.weight))
        if before.scale != after.scale:
            rescaled.append(ScaleChange(criterion_id=cid, old=before.scale, new=after.scale))
        elif before.guide != after.guide or before.anchors != after.anchors:
            guide.append(cid)
        if before.gating != after.gating:
            gating.append(GatingChange(criterion_id=cid, old=before.gating, new=after.gating))
        if before.title != after.title:
            retitled.append(cid)

    threshold = (
        ThresholdChange(old=old.pass_threshold, new=new.pass_threshold)
        if old.pass_threshold != new.pass_threshold
        else None
    )
    metadata = tuple(
        name for name in ("id", "title", "description") if getattr(old, name) != getattr(new, name)
    )
    return RubricDiff(
        added=tuple(CriterionRef.of(c) for c in new.criteria if c.id not in old_by_id),
        removed=tuple(CriterionRef.of(c) for c in old.criteria if c.id not in new_by_id),
        reweighted=tuple(reweighted),
        rescaled=tuple(rescaled),
        gating_changed=tuple(gating),
        threshold=threshold,
        guide_changed=tuple(guide),
        retitled=tuple(retitled),
        reordered=common != [cid for cid in new.criterion_ids if cid in old_by_id],
        metadata_changed=metadata,
    )
