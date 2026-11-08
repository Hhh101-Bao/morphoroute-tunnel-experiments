from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROFILE_DIRECTORY_NAMES = {
    "autodl_5090_32gb": (
        "RTX 5090 32GB AutoDL 加速正式实验配置",
        "autodl_5090_32gb",
    ),
    "local_6gb": ("RTX 4050 6GB 本机正式实验配置", "local_6gb"),
    "local_6gb_fallback": (
        "本机低显存预实验和流程验证配置",
        "local_6gb_fallback",
    ),
    "sop_full": ("完整 SOP 正式实验配置", "sop_full"),
}


def _existing_directory(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / names[-1]


def _existing_file(root: Path, *names: str) -> Path:
    for name in names:
        candidate = root / name
        if candidate.is_file():
            return candidate
    return root / names[-1]


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    experiments: Path
    tests: Path
    scripts: Path
    configs: Path
    environment: Path
    docs: Path
    verification_outputs: Path
    source: Path
    readme: Path
    pyproject: Path

    def script_file(self, *names: str) -> Path:
        return _existing_file(self.scripts, *names)

    def profile_config_dir(self, profile: str) -> Path:
        names = PROFILE_DIRECTORY_NAMES.get(profile, (profile,))
        return _existing_directory(self.configs, *names)

    @classmethod
    def discover(cls, root: str | Path) -> "ProjectLayout":
        project_root = Path(root).resolve(strict=False)
        return cls(
            root=project_root,
            experiments=_existing_directory(
                project_root,
                "19 个独立实验运行入口",
                "experiments",
            ),
            tests=_existing_directory(
                project_root,
                "61 项自动化测试",
                "tests",
            ),
            scripts=_existing_directory(
                project_root,
                "工程辅助运行脚本",
                "scripts",
            ),
            configs=_existing_directory(
                project_root,
                "三种运行档位的实验配置",
                "configs",
            ),
            environment=_existing_directory(
                project_root,
                "WSL2 与 Python 依赖文件",
                "environment",
            ),
            docs=_existing_directory(
                project_root,
                "中文实验说明和实验—输出对照表",
                "docs",
            ),
            verification_outputs=_existing_directory(
                project_root,
                "已生成的真实数据审计验证结果",
                "verification_outputs",
            ),
            source=_existing_directory(project_root, "src"),
            readme=_existing_file(
                project_root,
                "工程总说明与快速开始.md",
                "README.md",
            ),
            pyproject=_existing_file(
                project_root,
                "Python 工程及依赖配置.toml",
                "pyproject.toml",
            ),
        )
