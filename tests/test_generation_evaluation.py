from scripts.run_generation_evaluation import retry_call


def test_retry_call_retries_empty_then_succeeds():
    values = iter(["", "answer"])
    value, attempts, errors = retry_call(lambda: next(values), 3)
    assert value == "answer"
    assert attempts == 2
    assert len(errors) == 1


def test_retry_call_reports_exhaustion():
    value, attempts, errors = retry_call(lambda: "", 3)
    assert value is None
    assert attempts == 3
    assert len(errors) == 3
