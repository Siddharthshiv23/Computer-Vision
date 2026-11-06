"""The architecture diagram must come from the code, not from memory."""
from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("gen_architecture", ROOT / "scripts" / "gen_architecture.py")
gen = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gen)


def tree(tmp_path):
    (tmp_path / "isp" / "classical").mkdir(parents=True)
    (tmp_path / "app").mkdir()
    (tmp_path / "isp" / "__init__.py").write_text("")
    (tmp_path / "isp" / "types.py").write_text("import numpy\n")
    (tmp_path / "isp" / "classical" / "__init__.py").write_text("")
    (tmp_path / "isp" / "classical" / "demosaic.py").write_text("from isp.types import RawImage\n")
    (tmp_path / "isp" / "classical" / "pipeline.py").write_text("from isp.classical import demosaic\nfrom .demosaic import x\n")
    (tmp_path / "app" / "__init__.py").write_text("")
    (tmp_path / "app" / "adapter.py").write_text('SPEC = ("isp.classical.pipeline", "isp.not_landed")\n')
    return tmp_path


def test_lists_real_modules_only(tmp_path):
    modules = gen.discover(tree(tmp_path))
    assert set(modules) == {"isp.types", "isp.classical.demosaic", "isp.classical.pipeline", "app.adapter"}


def test_finds_import_edges(tmp_path):
    found = gen.edges(tree(tmp_path))
    assert ("isp.classical.demosaic", "isp.types", False) in found
    assert ("isp.classical.pipeline", "isp.classical.demosaic", False) in found


def test_string_references_are_dynamic_edges_and_unlanded_names_are_dropped(tmp_path):
    found = gen.edges(tree(tmp_path))
    assert ("app.adapter", "isp.classical.pipeline", True) in found
    assert not any(target == "isp.not_landed" for _, target, _ in found)


def test_mermaid_groups_modules_by_package(tmp_path):
    text = gen.mermaid(tree(tmp_path))
    assert text.startswith("flowchart")
    assert 'subgraph isp_classical["isp/classical"]' in text
    assert "isp_classical_pipeline --> isp_classical_demosaic" in text
    assert "app_adapter -.-> isp_classical_pipeline" in text


def test_update_replaces_only_the_generated_block(tmp_path):
    doc = tmp_path / "architecture.md"
    doc.write_text(f"before\n{gen.BEGIN}\nold\n{gen.END}\nafter\n")
    gen.update(doc, "new")
    text = doc.read_text()
    assert "old" not in text and "new" in text
    assert text.startswith("before\n") and text.endswith("after\n")
