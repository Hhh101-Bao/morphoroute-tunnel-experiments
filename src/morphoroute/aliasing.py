from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType


def _attach_to_parent(qualified_name: str, module: ModuleType) -> None:
    parent_name, _, child_name = qualified_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, child_name, module)


def install_package_alias(
    qualified_name: str,
    directory: str | Path,
) -> ModuleType:
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    package_directory = Path(directory)
    initializer = package_directory / "__init__.py"
    if not initializer.is_file():
        raise FileNotFoundError(
            f"package alias source lacks __init__.py: {package_directory}"
        )
    spec = importlib.util.spec_from_file_location(
        qualified_name,
        initializer,
        submodule_search_locations=[str(package_directory)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build package alias: {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    _attach_to_parent(qualified_name, module)
    return module


def install_module_alias(
    qualified_name: str,
    source_file: str | Path,
) -> ModuleType:
    existing = sys.modules.get(qualified_name)
    if existing is not None:
        return existing
    path = Path(source_file)
    if not path.is_file():
        raise FileNotFoundError(f"module alias source does not exist: {path}")
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build module alias: {qualified_name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(qualified_name, None)
        raise
    _attach_to_parent(qualified_name, module)
    return module
