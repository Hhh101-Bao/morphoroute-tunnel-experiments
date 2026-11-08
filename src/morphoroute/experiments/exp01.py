from __future__ import annotations

from morphoroute.core.config import ExperimentConfig
from morphoroute.core.io import atomic_write_json
from morphoroute.data.audit import audit_records, count_unresolved_duplicates
from morphoroute.data.manifest import load_manifest
from morphoroute.visualization.plots import plot_class_distribution
from morphoroute.visualization.tables import save_result_bundle

from .common import begin, dry_plan, finish, write_csv


def execute(config: ExperimentConfig, *, dry_run: bool = False):
    plan = dry_plan(
        config,
        [
            "读取唯一权威 split_manifest.csv",
            "核验文件对应、划分数量、group 隔离和五类统计",
            "生成 JSON/CSV/XLSX 审计结果和类别分布图",
        ],
        comparison_scope="data_integrity",
    )
    if dry_run:
        return plan
    directory = begin(config)
    manifest_path = config.data_root / "metadata" / "split_manifest.csv"
    records = load_manifest(manifest_path, config.data_root)
    verify_hashes = bool(config.parameters.get("verify_hashes", False))
    unresolved_duplicates = count_unresolved_duplicates(
        config.data_root / "metadata" / "duplicate_audit.csv"
    )
    report = audit_records(
        records,
        verify_hashes=verify_hashes,
        unresolved_duplicates=unresolved_duplicates,
    )
    atomic_write_json(directory / "data_audit.json", report.to_dict())
    distribution_rows = [
        {
            "split": split,
            "class_name": class_name,
            "positive_images": count,
            "instances": report.class_instances.get(split, {}).get(class_name, 0),
        }
        for split, values in report.class_positive_images.items()
        for class_name, count in values.items()
    ]
    save_result_bundle(
        distribution_rows,
        directory / "class_distribution_verified",
    )
    write_csv(
        directory / "split_overlap_check.csv",
        report.cross_split_groups,
        ("group_id", "splits"),
    )
    write_csv(
        directory / "dataset_hashes.csv",
        [
            {
                "record_id": record.record_id,
                "image_sha256": record.image_sha256,
                "annotation_sha256": record.annotation_sha256,
                "fully_rehashed": verify_hashes,
            }
            for record in records
        ],
    )
    try:
        plot_class_distribution(
            distribution_rows,
            directory / "class_distribution",
        )
    except RuntimeError:
        pass
    status = "completed" if report.passed else "failed_audit"
    return finish(
        config,
        directory,
        status=status,
        produced_outputs=plan["planned_outputs"],
        notes=(f"verify_hashes={verify_hashes}",),
    )
