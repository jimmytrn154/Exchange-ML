from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from audit_e1_e5_evidence import execution_signature, require, sha


class EvidenceAuditTests(unittest.TestCase):
    def test_e2_signature_binds_mode_and_matches_producer_contract(self):
        manifest = {"provenance": {"config_sha256": "fixed"}, "smoke_only": False}
        expected = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
        self.assertEqual(execution_signature("E2", manifest), expected)
        self.assertNotEqual(execution_signature("E2", manifest), execution_signature("E1", manifest))
        self.assertNotEqual(execution_signature("E2", manifest),
                            execution_signature("E2", {**manifest, "smoke_only": True}))

    def test_hash_audit_rejects_same_length_corruption(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.csv"
            path.write_bytes(b"original")
            digest = sha(path)
            require(path, digest)
            path.write_bytes(b"corrupt!")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                require(path, digest)


if __name__ == "__main__":
    unittest.main()
