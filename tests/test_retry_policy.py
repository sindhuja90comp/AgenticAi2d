from agentic_ai_2d.retry_policy import FailureClass, RepairBudgetKey, RetryRepairEnforcer


def key(input_digest: str = "sha256:input") -> RepairBudgetKey:
    return RepairBudgetKey("proj_12345678", 1, "render", input_digest)


def test_infrastructure_retries_use_exponential_backoff_without_repair_budget() -> None:
    enforcer = RetryRepairEnforcer()

    decisions = [enforcer.handle_failure(key(), FailureClass.INFRASTRUCTURE) for _ in range(4)]

    assert [decision.retry for decision in decisions] == [True, True, True, False]
    assert [decision.delay_seconds for decision in decisions] == [1, 2, 4, None]
    assert not any(decision.repair_attempted for decision in decisions)


def test_schema_repair_is_limited_to_one_attempt() -> None:
    enforcer = RetryRepairEnforcer()

    first = enforcer.handle_failure(key(), FailureClass.SCHEMA)
    second = enforcer.handle_failure(key(), FailureClass.SCHEMA)

    assert first.retry and first.repair_attempted
    assert not second.retry and second.requires_user_action
    assert not second.enter_qa_failed


def test_content_quality_repair_exhaustion_enters_qa_failed() -> None:
    enforcer = RetryRepairEnforcer()

    first = enforcer.handle_failure(key(), FailureClass.CONTENT_QUALITY)
    second = enforcer.handle_failure(key(), FailureClass.CONTENT_QUALITY)

    assert first.retry and first.repair_attempted
    assert not second.retry and second.enter_qa_failed and second.requires_user_action


def test_safety_never_retries() -> None:
    decision = RetryRepairEnforcer().handle_failure(key(), FailureClass.SAFETY)

    assert not decision.retry
    assert decision.requires_user_action
