"""Bounded retry and repair decisions for durable workflow activities."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class FailureClass(StrEnum):
    INFRASTRUCTURE = "infrastructure"
    SCHEMA = "schema"
    CONTENT_QUALITY = "content_quality"
    SAFETY = "safety"


@dataclass(frozen=True)
class RepairBudgetKey:
    """Stable scope for repair attempts; infrastructure retries are separate."""

    project_id: str
    project_version: int
    workflow_step: str
    input_digest: str

    def __post_init__(self) -> None:
        if self.project_version < 1:
            raise ValueError("project_version must be at least 1")
        if not all((self.project_id, self.workflow_step, self.input_digest)):
            raise ValueError("repair budget key fields must not be empty")

    @property
    def value(self) -> str:
        return ":".join(
            (self.project_id, str(self.project_version), self.workflow_step, self.input_digest)
        )


@dataclass(frozen=True)
class RetryDecision:
    retry: bool
    delay_seconds: int | None
    repair_attempted: bool
    enter_qa_failed: bool
    requires_user_action: bool


class RetryRepairEnforcer:
    """Track bounded infrastructure retries and content-specific repair budgets."""

    MAX_INFRASTRUCTURE_RETRIES = 3
    MAX_SCHEMA_REPAIRS = 1
    MAX_CONTENT_REPAIRS = 1

    def __init__(self) -> None:
        self._infrastructure_retries: dict[RepairBudgetKey, int] = {}
        self._schema_repairs: dict[RepairBudgetKey, int] = {}
        self._content_repairs: dict[RepairBudgetKey, int] = {}

    def handle_failure(self, key: RepairBudgetKey, failure_class: FailureClass) -> RetryDecision:
        """Return the only permitted response to a classified workflow failure."""
        if failure_class is FailureClass.INFRASTRUCTURE:
            retries = self._infrastructure_retries.get(key, 0)
            if retries >= self.MAX_INFRASTRUCTURE_RETRIES:
                return RetryDecision(False, None, False, False, True)
            retries += 1
            self._infrastructure_retries[key] = retries
            return RetryDecision(True, 2 ** (retries - 1), False, False, False)

        if failure_class is FailureClass.SCHEMA:
            repairs = self._schema_repairs.get(key, 0)
            if repairs >= self.MAX_SCHEMA_REPAIRS:
                return RetryDecision(False, None, False, False, True)
            self._schema_repairs[key] = repairs + 1
            return RetryDecision(True, None, True, False, False)

        if failure_class is FailureClass.CONTENT_QUALITY:
            repairs = self._content_repairs.get(key, 0)
            if repairs >= self.MAX_CONTENT_REPAIRS:
                return RetryDecision(False, None, False, True, True)
            self._content_repairs[key] = repairs + 1
            return RetryDecision(True, None, True, False, False)

        return RetryDecision(False, None, False, False, True)
