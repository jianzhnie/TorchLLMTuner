"""Inspect HF configuration/class availability without loading model weights.

This is a source-availability check, not a training or NPU capability test.
Remote model code is never executed by this command.
"""

from __future__ import annotations

import argparse
import json
import sys

import transformers
from transformers import AutoConfig, PretrainedConfig
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES


def inspect_model(
    model: str,
    revision: str | None,
    local_only: bool,
    expected_model_types: tuple[str, ...] = (),
    expected_top_level_type: str | None = None,
) -> dict:
    report = {
        "model": model,
        "revision": revision,
        "transformers": transformers.__version__,
        "native_causal_lm_available": False,
        "training_verified": False,
        "npu_verified": False,
        "expected_model_types": list(expected_model_types),
        "expected_top_level_type": expected_top_level_type,
    }
    try:
        # Read declarations before AutoConfig can infer a class from a path name
        # or reject unknown remote-code architectures. A directory name is not
        # evidence that its contents belong to the requested model family.
        raw, _ = PretrainedConfig.get_config_dict(
            model, revision=revision, local_files_only=local_only
        )
        text_raw = raw.get("text_config", raw)
        if not isinstance(text_raw, dict):
            raise ValueError("text_config must be a configuration object")
        model_type = text_raw.get("model_type")
        report.update(
            declared_model_type=raw.get("model_type"),
            model_type=model_type,
            advertised_architectures=text_raw.get("architectures"),
            advertised_auto_map=raw.get("auto_map"),
            resolved_revision=raw.get("_commit_hash"),
        )
        if (
            expected_top_level_type is not None
            and raw.get("model_type") != expected_top_level_type
        ):
            report["error"] = (
                "Model identity mismatch: expected top-level "
                f"{expected_top_level_type!r}, "
                f"but configuration declares {raw.get('model_type')!r}."
            )
            return report
        if expected_model_types and model_type not in expected_model_types:
            report["error"] = (
                f"Model identity mismatch: expected {expected_model_types}, "
                f"but configuration declares {model_type!r}. "
                "Use the correct official configuration and model source; "
                "renaming a directory does not change its architecture."
            )
            return report
        config = AutoConfig.from_pretrained(
            model,
            revision=revision,
            local_files_only=local_only,
            trust_remote_code=False,
        )
    except (OSError, ValueError, ImportError) as error:
        report["error"] = f"{type(error).__name__}: {error}"
        return report
    text = getattr(config, "text_config", config)
    model_type = text.model_type
    native_class = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.get(model_type)
    report.update(
        model_type=model_type,
        config_class=type(text).__name__,
        advertised_architectures=getattr(text, "architectures", None),
        native_causal_lm_class=native_class,
        native_causal_lm_available=native_class is not None,
    )
    if native_class is None:
        report["error"] = (
            "No native AutoModelForCausalLM mapping for this text configuration. "
            "A reviewed, pinned official HF training class is required; "
            "another architecture must not be substituted."
        )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--expected-model-type",
        action="append",
        default=[],
        help="Require a declared text model type (repeat to allow alternatives)",
    )
    parser.add_argument("--expected-top-level-type")
    args = parser.parse_args()
    report = inspect_model(
        args.model,
        args.revision,
        args.local_files_only,
        tuple(args.expected_model_type),
        args.expected_top_level_type,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["native_causal_lm_available"] else 2


if __name__ == "__main__":
    sys.exit(main())
