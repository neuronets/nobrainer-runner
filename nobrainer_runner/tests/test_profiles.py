"""Unit tests for the profile loader."""

from __future__ import annotations

import pytest
import yaml

from nobrainer_runner.profiles import load_profile


def _write_profile(tmp_path, name: str, data: dict) -> None:
    profiles_dir = tmp_path / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)
    (profiles_dir / f"{name}.yaml").write_text(yaml.dump(data))


class TestLoadProfile:
    def test_valid_slurm_profile(self, tmp_path):
        data = {"backend": "slurm", "defaults": {"partition": "gpu"}}
        _write_profile(tmp_path, "slurm_gpu", data)
        profile = load_profile("slurm_gpu", profiles_dir=tmp_path / "profiles")
        assert profile["backend"] == "slurm"
        assert profile["defaults"]["partition"] == "gpu"

    def test_valid_aws_profile(self, tmp_path):
        data = {"backend": "aws", "defaults": {"instance_type": "p3.2xlarge"}}
        _write_profile(tmp_path, "aws_gpu", data)
        profile = load_profile("aws_gpu", profiles_dir=tmp_path / "profiles")
        assert profile["backend"] == "aws"

    def test_valid_gcp_profile(self, tmp_path):
        data = {"backend": "gcp", "defaults": {"instance_type": "n1-highmem-4"}}
        _write_profile(tmp_path, "gcp_gpu", data)
        profile = load_profile("gcp_gpu", profiles_dir=tmp_path / "profiles")
        assert profile["backend"] == "gcp"

    def test_missing_profile_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_profile("nonexistent", profiles_dir=tmp_path)

    def test_missing_backend_raises(self, tmp_path):
        data = {"defaults": {"partition": "gpu"}}
        _write_profile(tmp_path, "bad_profile", data)
        with pytest.raises(ValueError, match="backend"):
            load_profile("bad_profile", profiles_dir=tmp_path / "profiles")

    def test_missing_partition_slurm_raises(self, tmp_path):
        data = {"backend": "slurm", "defaults": {}}
        _write_profile(tmp_path, "bad_slurm", data)
        with pytest.raises(ValueError, match="partition"):
            load_profile("bad_slurm", profiles_dir=tmp_path / "profiles")

    def test_missing_instance_type_aws_raises(self, tmp_path):
        data = {"backend": "aws", "defaults": {}}
        _write_profile(tmp_path, "bad_aws", data)
        with pytest.raises(ValueError, match="instance_type"):
            load_profile("bad_aws", profiles_dir=tmp_path / "profiles")

    def test_unknown_backend_raises(self, tmp_path):
        data = {"backend": "kubernetes", "defaults": {}}
        _write_profile(tmp_path, "bad_backend", data)
        with pytest.raises(ValueError, match="unknown backend"):
            load_profile("bad_backend", profiles_dir=tmp_path / "profiles")
