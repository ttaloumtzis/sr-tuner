"""Tests for models API — version listing and deletion."""

import json
import pytest

from sr_engine.api.routes import models as models_module
from sr_engine.workspace import Workspace


@pytest.fixture
def ws_with_versions(tmp_path):
    """Workspace with a model instance and two versions."""
    ws = Workspace(tmp_path / "workspace")
    ws.init()
    ws.create_model_instance("m1", {"name": "rrdb_esrgan", "scale": 4})
    v_dir = tmp_path / "workspace" / "models" / "m1" / "versions"
    (v_dir / "v1").mkdir()
    (v_dir / "v1" / "model.pt").write_text("dummy")
    (v_dir / "v1" / "version.json").write_text(
        json.dumps({"run": "run_001", "timestamp": 1000}),
        encoding="utf-8",
    )
    (v_dir / "v2").mkdir()
    (v_dir / "v2" / "model.pt").write_text("dummy2")
    return ws


class TestModelsApi:
    @pytest.fixture(autouse=True)
    def _init_ws(self, ws_with_versions):
        self.ws = ws_with_versions
        yield

    @pytest.mark.anyio
    async def test_list_versions(self):
        versions = await models_module.instance_versions("m1", self.ws)
        assert len(versions) == 2
        assert versions[0]["tag"] == "v1"
        assert versions[0]["has_weights"] is True
        assert versions[0]["metadata"] is not None
        assert versions[0]["metadata"]["run"] == "run_001"
        assert versions[1]["tag"] == "v2"
        assert versions[1]["has_weights"] is True

    @pytest.mark.anyio
    async def test_list_versions_404(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await models_module.instance_versions("nonexistent", self.ws)
        assert exc.value.status_code == 404

    @pytest.mark.anyio
    async def test_delete_version(self):
        result = await models_module.delete_version("m1", "v1", self.ws)
        assert result == {"deleted": "v1"}
        versions = self.ws.list_model_versions("m1")
        assert [v["tag"] for v in versions] == ["v2"]

    @pytest.mark.anyio
    async def test_delete_version_404_instance(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await models_module.delete_version("nonexistent", "v1", self.ws)
        assert exc.value.status_code == 404

    @pytest.mark.anyio
    async def test_delete_version_404_version(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await models_module.delete_version("m1", "v99", self.ws)
        assert exc.value.status_code == 404

    @pytest.mark.anyio
    async def test_delete_version_400_invalid_tag(self):
        from fastapi import HTTPException
        with pytest.raises(HTTPException) as exc:
            await models_module.delete_version("m1", "../etc", self.ws)
        assert exc.value.status_code == 400