"""MorphoRoute-Tunnel experiment toolkit."""

from pathlib import Path

from .aliasing import install_module_alias, install_package_alias


__version__ = "0.1.0"


def _install_chinese_layout_aliases() -> None:
    package_root = Path(__file__).resolve().parent
    package_directories = {
        "core": "配置、实验注册、运行记录和 D4 防泄漏组件",
        "data": "数据清单、审计、标注转换和数据导出组件",
        "models": "MorphoRoute 模型、专家、路由器、损失和 PCGrad",
        "metrics": "分割、结构、可靠性和量化指标",
        "statistics": "Bootstrap、效应量和 Holm 多重比较校正",
        "visualization": "实验表格及 PNGPDF 可视化图生成",
        "runtime": "训练适配、断点恢复和资源记录",
        "evaluation": "统一评价、预测格式和资源指标",
        "mmdet": "MMDetection 配置与扩展组件",
        "training": "三阶段训练、恢复和诊断组件",
        "analysis": "机制、量化和统计分析组件",
    }
    resolved_directories: dict[str, Path] = {}
    for alias, chinese_name in package_directories.items():
        standard = package_root / alias
        renamed = package_root / chinese_name
        if standard.is_dir():
            resolved_directories[alias] = standard
            continue
        if renamed.is_dir():
            install_package_alias(f"{__name__}.{alias}", renamed)
            resolved_directories[alias] = renamed

    renamed_modules = {
        ("core", "d4_guard"): "guard.py",
        ("runtime", "checkpoint"): "checkpoints.py",
    }
    for (package_alias, module_alias), filename in renamed_modules.items():
        directory = resolved_directories.get(package_alias)
        if directory is None:
            continue
        standard_module = directory / f"{module_alias}.py"
        renamed_module = directory / filename
        if not standard_module.is_file() and renamed_module.is_file():
            install_module_alias(
                f"{__name__}.{package_alias}.{module_alias}",
                renamed_module,
            )


_install_chinese_layout_aliases()
