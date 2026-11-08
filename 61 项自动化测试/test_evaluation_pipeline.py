from __future__ import annotations

import sys
import tempfile
import unittest
from dataclasses import fields, replace
from pathlib import Path
from unittest import mock

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _perfect_bundle():
    from morphoroute.evaluation.schema import (
        PredictionBundle,
        PredictionImage,
        PredictionInstance,
    )
    crack = np.zeros((12, 12), dtype=bool)
    crack[5, 2:10] = True
    area = np.zeros((12, 12), dtype=bool)
    area[2:7, 3:9] = True
    targets = (
        PredictionInstance(
            class_name="crack",
            score=1.0,
            bbox=(2, 5, 8, 1),
            mask=crack,
        ),
        PredictionInstance(
            class_name="spalling",
            score=1.0,
            bbox=(3, 2, 6, 5),
            mask=area,
        ),
    )
    predictions = tuple(
        PredictionInstance(
            class_name=item.class_name,
            score=0.99,
            bbox=item.bbox,
            mask=item.mask.copy(),
        )
        for item in targets
    )
    return PredictionBundle(
        experiment_id="TEST",
        split="D3",
        class_names=("crack", "spalling"),
        images=(
            PredictionImage(
                image_id="image-1",
                group_id="group-1",
                width=12,
                height=12,
                predictions=predictions,
                targets=targets,
            ),
        ),
    )


def _encoded_bundle(bundle):
    from morphoroute.evaluation.mask_codec import encode_mask

    def encoded_instance(instance):
        return replace(instance, mask=encode_mask(instance.mask))

    return replace(
        bundle,
        images=tuple(
            replace(
                image,
                predictions=tuple(
                    encoded_instance(instance)
                    for instance in image.predictions
                ),
                targets=tuple(
                    encoded_instance(instance) for instance in image.targets
                ),
            )
            for image in bundle.images
        ),
    )


class EvaluationPipelineTests(unittest.TestCase):
    def test_perfect_prediction_produces_perfect_scores(self):
        from morphoroute.evaluation.pipeline import evaluate_bundle

        result = evaluate_bundle(_perfect_bundle())

        self.assertAlmostEqual(result.mask_ap50, 1.0)
        self.assertAlmostEqual(result.crack_cldice, 1.0)
        self.assertAlmostEqual(result.crack_break_rate, 0.0)
        self.assertAlmostEqual(result.crack_component_error, 0.0)
        self.assertAlmostEqual(result.area_boundary_f1, 1.0)
        self.assertAlmostEqual(result.area_hd95, 0.0)
        self.assertAlmostEqual(result.area_relative_error, 0.0)

    def test_writer_creates_tables_json_and_visualizations(self):
        from morphoroute.evaluation.pipeline import (
            evaluate_bundle,
            write_evaluation_outputs,
        )

        with tempfile.TemporaryDirectory() as temporary:
            outputs = write_evaluation_outputs(
                evaluate_bundle(_perfect_bundle()),
                _perfect_bundle(),
                Path(temporary),
            )

            self.assertTrue(outputs.metrics_csv.is_file())
            self.assertTrue(outputs.metrics_xlsx.is_file())
            self.assertTrue(outputs.summary_json.is_file())
            self.assertTrue(outputs.overlay_png.is_file())
            self.assertTrue(outputs.overlay_pdf.is_file())

    def test_dense_and_rle_evaluation_are_numerically_identical(self):
        from morphoroute.evaluation.pipeline import (
            EvaluationResult,
            evaluate_bundle,
        )

        dense_result = evaluate_bundle(_perfect_bundle())
        encoded_result = evaluate_bundle(_encoded_bundle(_perfect_bundle()))

        for field in fields(EvaluationResult):
            dense_value = getattr(dense_result, field.name)
            encoded_value = getattr(encoded_result, field.name)
            if isinstance(dense_value, float):
                self.assertAlmostEqual(dense_value, encoded_value, places=12)
            else:
                self.assertEqual(dense_value, encoded_value)

    def test_encoded_masks_decode_once_per_matched_side(self):
        from morphoroute.evaluation import pipeline
        from morphoroute.evaluation.mask_codec import decode_mask

        encoded = _encoded_bundle(_perfect_bundle())
        with mock.patch.object(
            pipeline,
            "decode_mask",
            wraps=decode_mask,
            create=True,
        ) as decoder:
            pipeline.evaluate_bundle(encoded)

        self.assertEqual(decoder.call_count, 4)


if __name__ == "__main__":
    unittest.main()
