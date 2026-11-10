"""CPU artifact contracts; synthetic wheels do not prove native execution."""

import base64
import csv
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.build_ascendc_compat_wheel import (
    BASE_VERSION,
    FORK_VERSION,
    MARKER,
    build,
    repack,
)
from scripts.diagnose_ascendc_runtime import inspect_site
from scripts.diagnose_autofuse_ir import duplicate_fields


class AscendcArtifactTests(unittest.TestCase):
    def test_duplicate_api_fields_reject_within_the_same_struct(self):
        duplicate = """
BEGIN_TILING_DATA_DEF_T(Result1)
TILING_DATA_FIELD_DEF_T_STRUCT(PadTiling, sum_0_pad_tilingData_0);
TILING_DATA_FIELD_DEF_T_STRUCT(PadTiling, sum_0_pad_tilingData_0);
END_TILING_DATA_DEF_T;
"""
        self.assertEqual(
            duplicate_fields(duplicate),
            [{"struct": "Result1", "field": "sum_0_pad_tilingData_0"}],
        )
        single = duplicate.replace(
            "TILING_DATA_FIELD_DEF_T_STRUCT(PadTiling, sum_0_pad_tilingData_0);",
            "TILING_DATA_FIELD_DEF_T_STRUCT(PadTiling, distinct);",
            1,
        )
        self.assertEqual(duplicate_fields(single), [])
        separate_structs = single + single.replace("Result1", "Result2")
        self.assertEqual(duplicate_fields(separate_structs), [])

    def test_repack_is_reproducible_and_record_matches_every_member(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = root / "official.whl"
            with zipfile.ZipFile(original, "w") as archive:
                archive.writestr(
                    f"torch_npu-{BASE_VERSION}.dist-info/METADATA",
                    f"Name: torch_npu\nVersion: {BASE_VERSION}\n",
                )
                archive.writestr(
                    "torch_npu/version.py", f"__version__ = '{BASE_VERSION}'\n"
                )
                archive.writestr("torch_npu/lib/sample.so", bytes(range(256)) * 100)
                archive.writestr(
                    f"torch_npu-{BASE_VERSION}.dist-info/RECORD", "old record"
                )
            outputs = [root / "one.whl", root / "two.whl"]
            marker = {"formal_repository_acceptance": False}
            for output in outputs:
                repack(original, output, {}, marker)
            self.assertEqual(outputs[0].read_bytes(), outputs[1].read_bytes())
            with zipfile.ZipFile(outputs[0]) as archive:
                record = f"torch_npu-{FORK_VERSION}.dist-info/RECORD"
                rows = list(csv.reader(io.StringIO(archive.read(record).decode())))
                self.assertEqual({row[0] for row in rows}, set(archive.namelist()))
                for name, digest, size in rows:
                    if name == record:
                        self.assertEqual((digest, size), ("", ""))
                        continue
                    data = archive.read(name)
                    expected = base64.urlsafe_b64encode(hashlib.sha256(data).digest())
                    self.assertEqual(digest, "sha256=" + expected.rstrip(b"=").decode())
                    self.assertEqual(int(size), len(data))
                self.assertEqual(json.loads(archive.read(MARKER)), marker)
            with self.assertRaises(FileExistsError):
                repack(original, outputs[0], {}, marker)

    def test_unpinned_wheel_rejects_before_creating_output(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            wheel = root / "wrong.whl"
            wheel.write_bytes(b"unrecognized input")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                build(wheel, root / "output")
            self.assertFalse((root / "output").exists())

    def test_missing_runtime_and_wrong_environment_fail_without_lowering_claim(self):
        with tempfile.TemporaryDirectory() as temporary:
            result = inspect_site(Path(temporary), {})
            self.assertFalse(result["source_capability_ready"])
            self.assertFalse(result["runtime_lowering_proved"])
            self.assertFalse(result["native_kernel_execution_proved"])
            self.assertFalse(result["formal_repository_acceptance"])
            self.assertGreaterEqual(len(result["errors"]), 5)


if __name__ == "__main__":
    unittest.main()
