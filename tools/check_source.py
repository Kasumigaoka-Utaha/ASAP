"""Dependency-free syntax, local import, and documentation link checks.

Run from any directory: python tools/check_source.py
This validates source consistency; it does not run model training.
"""

import ast
from pathlib import Path
import re
import sys
from urllib.parse import unquote


ROOT = Path(__file__).resolve().parents[1]
COMPONENTS = (ROOT / "ASAP-3DBPP", ROOT / "ASAP-Routing")


def sources(directory):
    return sorted(p for p in directory.rglob("*.py") if not any(
        part in {".git", "__pycache__", ".venv"} for part in p.parts
    ))


def resolve_module(base, module):
    target = base.joinpath(*module.split("."))
    for candidate in (target.with_suffix(".py"), target / "__init__.py"):
        if candidate.is_file():
            return candidate
    if target.is_dir():
        return target  # Namespace package without __init__.py.
    return None


def check():
    errors = []
    count = 0
    for component in COMPONENTS:
        if not component.is_dir():
            errors.append(f"Missing component: {component.name}")
            continue
        files = sources(component)
        local_names = {p.stem for p in files if p.parent == component}
        local_names.update(p.name for p in component.iterdir() if p.is_dir()
                           and not p.name.startswith((".", "__")))
        trees = {}
        for path in files:
            try:
                trees[path] = ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
            except (SyntaxError, UnicodeError) as exc:
                errors.append(f"{path.relative_to(ROOT)}: {exc}")
        count += len(files)
        for path, tree in trees.items():
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or not node.module or node.level:
                    continue
                name = node.module.split(".")[0]
                bases = (path.parent, component, component / "utils", component / "generator", component / "baselines")
                target = next((p for base in bases if (p := resolve_module(base, node.module))), None)
                if target is None:
                    if name in local_names:
                        errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: missing local module {node.module}")
                    continue
                target_tree = trees.get(target)
                if target_tree is None or any(isinstance(n, ast.ImportFrom) and any(
                        a.name == "*" for a in n.names) for n in ast.walk(target_tree)):
                    continue
                defined = {n.id for n in ast.walk(target_tree) if isinstance(n, ast.Name)
                           and isinstance(n.ctx, ast.Store)}
                defined.update(n.name for n in ast.walk(target_tree)
                               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)))
                for n in ast.walk(target_tree):
                    if isinstance(n, (ast.Import, ast.ImportFrom)):
                        defined.update(a.asname or a.name.split(".")[0] for a in n.names)
                for alias in node.names:
                    if alias.name != "*" and alias.name not in defined and not resolve_module(target.parent, alias.name):
                        errors.append(f"{path.relative_to(ROOT)}:{node.lineno}: {node.module} has no {alias.name}")
    docs = [ROOT / "README.md", ROOT / "THIRD_PARTY_NOTICES.md"]
    docs.extend((ROOT / "docs").glob("*.md"))
    docs.extend(p / "README.md" for p in COMPONENTS)
    for path in docs:
        if not path.exists():
            errors.append(f"Missing documentation: {path.relative_to(ROOT)}")
            continue
        for link in re.findall(r"\]\(([^)]+)\)", path.read_text(encoding="utf-8")):
            if re.match(r"[a-z]+:|#", link, re.I):
                continue
            target = unquote(link.split("#")[0])
            if target and not (path.parent / target).exists():
                errors.append(f"{path.relative_to(ROOT)}: broken link {link}")
    for error in errors:
        print(error, file=sys.stderr)
    print(f"Checked {count} Python files and {len(docs)} documentation files; {len(errors)} errors.")
    return bool(errors)


if __name__ == "__main__":
    sys.exit(check())
