"""Exercise compiled-output SDC and an early non-finite fault injection."""

import os

import torch

from examples.train_qwen3_30b_npu import qwen3_30b_npu_config
from llmtuner.parallel import model_compile
from llmtuner.trainer.trainer import Trainer


def main() -> None:
    inject = os.environ.get("LLMTUNER_SDC_INJECT_NAN") == "1"
    if inject:
        original = model_compile.wrap_sdc_forward

        def injecting_wrapper(module, name):
            if name == "decoder layer 0":
                target = (
                    "_compiled_call_impl"
                    if module._compiled_call_impl is not None
                    else "forward"
                )
                forward = getattr(module, target)

                def fault(*args, **kwargs):
                    output = forward(*args, **kwargs)
                    output[0, 0, 0] = float("nan")
                    return output

                setattr(module, target, fault)
            return original(module, name)

        model_compile.wrap_sdc_forward = injecting_wrapper
    config = qwen3_30b_npu_config()
    config.training.compile_config.enable_sdc = True
    torch.set_default_dtype(torch.bfloat16)
    try:
        Trainer(config).train()
    except FloatingPointError as error:
        assert inject and "decoder layer 0" in str(error)
        print(f"PASS compile SDC rejected injected fault: {error}", flush=True)
    else:
        assert not inject, "compile SDC accepted the injected NaN"
        print("PASS compile SDC finite training", flush=True)
    finally:
        if inject:
            model_compile.wrap_sdc_forward = original


if __name__ == "__main__":
    main()
