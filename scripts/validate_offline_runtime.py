"""Read wheel metadata without importing or installing target runtime packages."""

import argparse
from email.parser import Parser
from pathlib import Path
from zipfile import ZipFile

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet
from packaging.tags import compatible_tags, cpython_tags
from packaging.utils import canonicalize_name, parse_wheel_filename
from packaging.version import Version


def target_contract(python_version, glibc_version):
    """Conservative CPython minor-version floor and manylinux glibc baseline.

    No host marker or wheel tag is inherited. Kernel release/version markers
    require actual target facts and are rejected by the offline check below.
    Unversioned linux wheels and musllinux are outside this glibc contract.
    """
    major, minor = (int(part) for part in python_version.split("."))
    glibc_major, glibc_minor = (int(part) for part in glibc_version.split("."))
    if major != 3 or minor < 10 or glibc_major != 2 or glibc_minor < 17:
        raise ValueError("Target requires CPython 3.10+ and glibc 2.17+")
    platforms = [
        f"manylinux_2_{version}_aarch64" for version in range(glibc_minor, 16, -1)
    ] + ["manylinux2014_aarch64"]
    interpreter = f"cp{major}{minor}"
    tags = set(
        cpython_tags((major, minor), abis=[interpreter], platforms=platforms)
    ) | set(
        compatible_tags((major, minor), interpreter=interpreter, platforms=platforms)
    )
    environment = {
        "implementation_name": "cpython",
        "implementation_version": f"{python_version}.0",
        "os_name": "posix",
        "platform_machine": "aarch64",
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "python_full_version": f"{python_version}.0",
        "python_version": python_version,
        "sys_platform": "linux",
        "extra": "",
    }
    return environment, tags


def validate(
    paths,
    *,
    expected_transformers,
    python_version,
    transformers_root=None,
    glibc_version="2.28",
):
    distributions = {}
    environment, target_tags = target_contract(python_version, glibc_version)

    def add(metadata, source):
        name = canonicalize_name(metadata["Name"])
        if name in distributions:
            raise ValueError(f"Duplicate distribution: {name}")
        distributions[name] = (metadata, source)

    for path in paths:
        wheel_name, wheel_version, _, wheel_tags = parse_wheel_filename(Path(path).name)
        if not wheel_tags.intersection(target_tags):
            raise ValueError(f"Wheel tags incompatible with target: {path}")
        with ZipFile(path) as archive:
            entries = [
                name
                for name in archive.namelist()
                if name.endswith(".dist-info/METADATA") and name.count("/") == 1
            ]
            if len(entries) != 1:
                raise ValueError(f"Expected one wheel METADATA: {path}")
            metadata = Parser().parsestr(archive.read(entries[0]).decode())
            if (
                canonicalize_name(metadata["Name"]) != wheel_name
                or Version(metadata["Version"]) != wheel_version
            ):
                raise ValueError(f"Wheel filename/METADATA mismatch: {path}")
            add(metadata, str(path))
    if transformers_root:
        entries = list(
            Path(transformers_root).glob("transformers-*.dist-info/METADATA")
        )
        if len(entries) != 1:
            raise ValueError("Extracted runtime must contain one Transformers METADATA")
        add(Parser().parsestr(entries[0].read_text()), str(entries[0]))
    if "transformers" not in distributions:
        raise ValueError("Missing Transformers distribution")
    version = distributions["transformers"][0]["Version"]
    if not version.startswith(expected_transformers) or (
        expected_transformers == "5.22.2" and version != expected_transformers
    ):
        raise ValueError(f"Transformers {version} != expected {expected_transformers}")
    errors = []
    for name, (metadata, _) in distributions.items():
        requires_python = metadata.get("Requires-Python")
        if requires_python:
            if Version(environment["python_full_version"]) not in SpecifierSet(
                requires_python
            ):
                errors.append(f"{name} requires Python {requires_python}")
        for raw in metadata.get_all("Requires-Dist", []):
            requirement = Requirement(raw)
            if requirement.marker and any(
                variable in str(requirement.marker)
                for variable in ("platform_release", "platform_version")
            ):
                errors.append(
                    f"{name}: target kernel marker needs explicit facts: {raw}"
                )
                continue
            if requirement.marker and not requirement.marker.evaluate(environment):
                continue
            dependency = canonicalize_name(requirement.name)
            selected = distributions.get(dependency)
            if selected is None:
                errors.append(f"{name} requires missing {requirement}")
            elif Version(selected[0]["Version"]) not in requirement.specifier:
                errors.append(
                    f"{name} requires {requirement}; selected {selected[0]['Version']}"
                )
    if errors:
        raise ValueError("Offline dependency contract failed:\n" + "\n".join(errors))
    return {name: item[0]["Version"] for name, item in sorted(distributions.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-transformers", required=True)
    parser.add_argument("--python-version", default="3.12")
    parser.add_argument("--transformers-root")
    parser.add_argument("--glibc-version", default="2.28")
    parser.add_argument("wheels", nargs="+", type=Path)
    args = parser.parse_args()
    print(
        validate(
            args.wheels,
            expected_transformers=args.expected_transformers,
            python_version=args.python_version,
            transformers_root=args.transformers_root,
            glibc_version=args.glibc_version,
        )
    )


if __name__ == "__main__":
    main()
