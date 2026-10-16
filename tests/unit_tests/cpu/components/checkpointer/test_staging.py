"""Staging failures must finish both caller and upload futures."""

from concurrent.futures import Future, ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from llmtuner.components.checkpointer.distributed_checkpoint import CoordinatedStager


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("asynchronous", [False, True])
def test_staging_failures_finish_futures_without_breaking_callbacks(
    error_type, asynchronous
):
    failure = error_type("injected staging failure")
    source = Future()

    def stage(state):
        if asynchronous:
            return source
        raise failure

    stager = CoordinatedStager(SimpleNamespace(stage=stage), None)
    try:
        upload = stager.stage({})
    except BaseException as error:
        raise AssertionError("Staging failure bypassed completion futures") from error

    # Match PyTorch 2.14's callback boundary. Interrupts must not escape into
    # Future.set_exception and terminate the staging executor's worker.
    downstream = Future()

    def callback(result):
        try:
            result.result()
        except Exception as error:
            downstream.set_exception(error)
        else:
            downstream.set_result(None)

    upload.add_done_callback(callback)
    if asynchronous:
        with ThreadPoolExecutor(max_workers=1) as executor:
            try:
                executor.submit(source.set_exception, failure).result(timeout=5)
            except BaseException as error:
                raise AssertionError("Staging callback leaked its failure") from error
            assert executor.submit(lambda: "alive").result(timeout=5) == "alive"
    assert stager.completion.done()
    assert upload.done()
    assert downstream.done()
    with pytest.raises(error_type) as caught:
        stager.synchronize_staging()
    assert caught.value is failure
    upstream_error = downstream.exception()
    if isinstance(failure, Exception):
        assert upstream_error is failure
    else:
        assert isinstance(upstream_error, RuntimeError)
        assert upstream_error.__cause__ is failure
