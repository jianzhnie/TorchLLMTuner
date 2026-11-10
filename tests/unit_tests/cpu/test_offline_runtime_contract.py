"""Offline installer checks use metadata, never target imports or installation."""

import importlib.util
from pathlib import Path
from zipfile import ZipFile

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts/validate_offline_runtime.py"
SPEC = importlib.util.spec_from_file_location("offline_runtime_contract", SCRIPT)
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def wheel(
    tmp_path, name, version, requirements=(), python_requires=None, tag="py3-none-any"
):
    path = tmp_path / f"{name}-{version}-{tag}.whl"
    metadata = f"Name: {name}\nVersion: {version}\n"
    metadata += "".join(f"Requires-Dist: {item}\n" for item in requirements)
    if python_requires:
        metadata += f"Requires-Python: {python_requires}\n"
    with ZipFile(path, "w") as archive:
        archive.writestr(f"{name}-{version}.dist-info/METADATA", metadata)
        archive.writestr("vendor/unrelated.dist-info/METADATA", "Name: unrelated\n")
    return path


def validate(paths, **kwargs):
    return VALIDATOR.validate(
        paths, expected_transformers="5.22.2", python_version="3.12", **kwargs
    )


@pytest.mark.parametrize("hub,tokenizers", [("1.5.0", "0.23.3"), ("2.4.0", "0.22.2")])
def test_rejects_old_runtime_dependency_versions(tmp_path, hub, tokenizers):
    paths = [
        wheel(
            tmp_path,
            "transformers",
            "5.22.2",
            ("huggingface-hub>=1.31,<3", "tokenizers>=0.23.1,<0.24"),
        ),
        wheel(tmp_path, "huggingface_hub", hub),
        wheel(tmp_path, "tokenizers", tokenizers),
    ]
    with pytest.raises(ValueError, match="selected"):
        validate(paths)


def test_checks_transitive_dependencies_and_target_markers(tmp_path):
    paths = [
        wheel(tmp_path, "transformers", "5.22.2", ("huggingface-hub>=1.31",)),
        wheel(
            tmp_path,
            "huggingface_hub",
            "2.4.0",
            ("httpx2>=2; platform_machine == 'aarch64'",),
        ),
    ]
    with pytest.raises(ValueError, match="missing httpx2"):
        validate(paths)
    paths.append(wheel(tmp_path, "httpx2", "2.16.0"))
    assert validate(paths)["httpx2"] == "2.16.0"


def test_checks_every_selected_distribution_not_only_transformers(tmp_path):
    paths = [
        wheel(tmp_path, "transformers", "5.22.2"),
        wheel(tmp_path, "datasets", "3.6.0", ("fsspec<=2025.3.0",)),
        wheel(tmp_path, "fsspec", "2026.9.0"),
    ]
    with pytest.raises(ValueError, match="datasets requires"):
        validate(paths)


def test_rejects_python_contract_and_duplicates(tmp_path):
    path = wheel(tmp_path, "transformers", "5.22.2", python_requires=">=3.13")
    with pytest.raises(ValueError, match="requires Python"):
        validate([path])
    with pytest.raises(ValueError, match="Duplicate"):
        validate([path, path])


def test_default_version_is_exact_and_ignores_optional_extras(tmp_path):
    path = wheel(tmp_path, "transformers", "5.22.20")
    with pytest.raises(ValueError, match="expected 5.22.2"):
        validate([path])
    path = wheel(tmp_path, "transformers", "5.22.2", ("unused; extra == 'test'",))
    assert validate([path])["transformers"] == "5.22.2"


def test_explicit_extracted_legacy_metadata(tmp_path):
    metadata = tmp_path / "transformers-5.13.0.dist-info/METADATA"
    metadata.parent.mkdir()
    metadata.write_text("Name: transformers\nVersion: 5.13.0\n")
    assert VALIDATOR.validate(
        [],
        expected_transformers="5.13.",
        python_version="3.12",
        transformers_root=tmp_path,
    ) == {"transformers": "5.13.0"}


@pytest.mark.parametrize(
    "tag",
    [
        "cp312-cp312-manylinux_2_28_x86_64",
        "cp311-cp311-manylinux_2_28_aarch64",
        "cp313-abi3-manylinux_2_28_aarch64",
        "cp312-cp312t-manylinux_2_28_aarch64",
        "cp312-cp312-manylinux_2_29_aarch64",
        "cp312-cp312-linux_aarch64",
        "cp312-cp312-musllinux_1_2_aarch64",
        "cp312-cp312-win_arm64",
    ],
)
def test_rejects_incompatible_arch_interpreter_abi_and_libc(tmp_path, tag):
    path = wheel(tmp_path, "transformers", "5.22.2", tag=tag)
    with pytest.raises(ValueError, match="tags incompatible"):
        validate([path])


@pytest.mark.parametrize(
    "tag",
    [
        "cp312-cp312-manylinux_2_28_aarch64",
        "cp310-abi3-manylinux_2_17_aarch64",
        "cp39-abi3-manylinux2014_aarch64",
        "py2.py3-none-any",
    ],
)
def test_accepts_target_compatible_native_stable_abi_and_pure_python(tmp_path, tag):
    path = wheel(tmp_path, "transformers", "5.22.2", tag=tag)
    assert validate([path])["transformers"] == "5.22.2"


def test_markers_never_inherit_host_facts(tmp_path, monkeypatch):
    import platform

    monkeypatch.setattr(platform, "python_version", lambda: "3.99.8")
    monkeypatch.setattr(platform, "python_implementation", lambda: "PyPy")
    requirements = (
        "missing; implementation_version >= '3.13'",
        "missing; platform_python_implementation == 'PyPy'",
        "missing; os_name == 'nt'",
        "required; implementation_version == '3.12.0'",
    )
    path = wheel(tmp_path, "transformers", "5.22.2", requirements)
    with pytest.raises(ValueError, match="missing required"):
        validate([path])
    assert validate([path, wheel(tmp_path, "required", "1.0")])


@pytest.mark.parametrize("marker", ["platform_release", "platform_version"])
def test_unknown_kernel_markers_rejected_instead_of_using_host(tmp_path, marker):
    path = wheel(tmp_path, "transformers", "5.22.2", (f"dep; {marker} == 'x'",))
    with pytest.raises(ValueError, match="kernel marker needs explicit facts"):
        validate([path])


def test_filename_metadata_identity_must_match(tmp_path):
    path = wheel(tmp_path, "transformers", "5.22.2")
    wrong = path.with_name("transformers-5.22.3-py3-none-any.whl")
    path.rename(wrong)
    with pytest.raises(ValueError, match="filename/METADATA mismatch"):
        validate([wrong])
