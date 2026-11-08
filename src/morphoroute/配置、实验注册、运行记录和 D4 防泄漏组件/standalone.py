from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Sequence

from morphoroute.core.config import VALID_PROFILES


AUTHORITATIVE_DATASET_ROOT = Path(
    r"E:\桌面"
    r"\铁路隧道多病害实例分割_原始影像严格划分_v1"
)


@dataclass(frozen=True)
class EntrySettings:
    profile: str
    dataset_root: Path
    output_root: Path
    smoke_test: bool
    auto_run_prerequisites: bool
    resume: bool

    @classmethod
    def defaults(cls, project_root: Path | None = None) -> "EntrySettings":
        root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[3]
        )
        return cls(
            profile="local_6gb",
            dataset_root=AUTHORITATIVE_DATASET_ROOT,
            output_root=root / "outputs",
            smoke_test=False,
            auto_run_prerequisites=True,
            resume=True,
        )


@dataclass(frozen=True)
class ParsedEntryArguments:
    settings: EntrySettings
    dry_run: bool
    analysis_only: bool
    config_path: Path | None
    execution_stage: str | None

    def relay_tokens(self) -> list[str]:
        tokens = [
            "--profile",
            self.settings.profile,
            "--dataset-root",
            str(self.settings.dataset_root),
            "--output-root",
            str(self.settings.output_root),
            (
                "--smoke-test"
                if self.settings.smoke_test
                else "--no-smoke-test"
            ),
            (
                "--auto-run-prerequisites"
                if self.settings.auto_run_prerequisites
                else "--no-auto-run-prerequisites"
            ),
            "--resume" if self.settings.resume else "--no-resume",
        ]
        if self.config_path is not None:
            tokens.extend(("--config", str(self.config_path)))
        if self.analysis_only:
            tokens.append("--analysis-only")
        if self.execution_stage is not None:
            tokens.extend(("--stage", self.execution_stage))
        return tokens


def build_entry_parser(defaults: EntrySettings) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one railway-tunnel experiment as a standalone workflow."
    )
    parser.add_argument(
        "--profile",
        choices=tuple(sorted(VALID_PROFILES)),
        default=defaults.profile,
    )
    parser.add_argument("--dataset-root", type=Path, default=defaults.dataset_root)
    parser.add_argument("--output-root", type=Path, default=defaults.output_root)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("prefreeze", "external"),
        help="Internal EXP-12 stage; normally selected automatically.",
    )
    parser.add_argument(
        "--smoke-test",
        action=argparse.BooleanOptionalAction,
        default=defaults.smoke_test,
    )
    parser.add_argument(
        "--auto-run-prerequisites",
        action=argparse.BooleanOptionalAction,
        default=defaults.auto_run_prerequisites,
    )
    parser.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=defaults.resume,
    )
    return parser


def parse_entry_arguments(
    argv: Sequence[str] | None,
    defaults: EntrySettings,
) -> ParsedEntryArguments:
    arguments = build_entry_parser(defaults).parse_args(argv)
    settings = replace(
        defaults,
        profile=arguments.profile,
        dataset_root=arguments.dataset_root,
        output_root=arguments.output_root,
        smoke_test=arguments.smoke_test,
        auto_run_prerequisites=arguments.auto_run_prerequisites,
        resume=arguments.resume,
    )
    return ParsedEntryArguments(
        settings=settings,
        dry_run=arguments.dry_run,
        analysis_only=arguments.analysis_only,
        config_path=arguments.config,
        execution_stage=arguments.stage,
    )
