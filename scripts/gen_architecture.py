"""Generate the Mermaid module diagram in docs/architecture.md from the code itself.

Usage: python scripts/gen_architecture.py
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("isp", "app")
BEGIN = "<!-- BEGIN GENERATED ARCHITECTURE -->"
END = "<!-- END GENERATED ARCHITECTURE -->"


def discover(root: Path) -> dict[str, Path]:
    """Map dotted module name to file, leaving out package __init__ files."""
    found = {}
    for package in PACKAGES:
        for path in sorted((root / package).rglob("*.py")):
            if path.name != "__init__.py":
                found[".".join(path.relative_to(root).with_suffix("").parts)] = path
    return found


def _imports(tree: ast.AST, module: str) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parent = module.split(".")[: -node.level]
                base = ".".join([*parent, base] if base else parent)
            names.add(base)
            names.update(f"{base}.{alias.name}" for alias in node.names)
    return names


def edges(root: Path) -> list[tuple[str, str, bool]]:
    """Return (source, target, dynamic). Dynamic means referenced by name in a string."""
    modules = discover(root)
    found = set()
    for module, path in modules.items():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        static = (_imports(tree, module) & modules.keys()) - {module}
        strings = {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        dynamic = (strings & modules.keys()) - static - {module}
        found.update((module, target, False) for target in static)
        found.update((module, target, True) for target in dynamic)
    return sorted(found)


def _node(module: str) -> str:
    return module.replace(".", "_")


def mermaid(root: Path) -> str:
    groups: dict[str, list[str]] = {}
    for module in discover(root):
        groups.setdefault(module.rpartition(".")[0], []).append(module)
    lines = ["flowchart LR"]
    for package, members in sorted(groups.items()):
        lines.append(f'    subgraph {_node(package)}["{package.replace(".", "/")}"]')
        lines.extend(f'        {_node(m)}["{m.rpartition(".")[2]}"]' for m in members)
        lines.append("    end")
    for source, target, dynamic in edges(root):
        lines.append(f"    {_node(source)} {'-.->' if dynamic else '-->'} {_node(target)}")
    return "\n".join(lines)


def update(doc: Path, diagram: str) -> None:
    text = doc.read_text(encoding="utf-8")
    head, _, rest = text.partition(BEGIN)
    _, _, tail = rest.partition(END)
    doc.write_text(f"{head}{BEGIN}\n{diagram}\n{END}{tail}", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    update(ROOT / "docs" / "architecture.md", f"```mermaid\n{mermaid(ROOT)}\n```")
    print("Updated docs/architecture.md")
