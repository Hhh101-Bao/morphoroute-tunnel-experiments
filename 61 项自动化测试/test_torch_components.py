import importlib.util
import sys
import unittest
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is installed in WSL ENV-MAIN")
class TorchComponentTests(unittest.TestCase):
    def test_sparse_decoder_returns_query_masks_and_activation_counts(self):
        import torch

        from morphoroute.models.experts import SparseExpertDecoder

        decoder = SparseExpertDecoder(threshold=0.75)
        queries = torch.randn(1, 3, 8)
        line_features = torch.randn(1, 8, 4, 4)
        area_features = torch.randn(1, 8, 4, 4)
        route_logits = torch.tensor([[[4.0, 0.0], [0.0, 4.0], [0.0, 0.0]]])
        decoder.eval()
        masks, diagnostics = decoder(
            queries, line_features, area_features, route_logits
        )
        self.assertEqual(tuple(masks.shape), (1, 3, 4, 4))
        self.assertEqual(diagnostics["active_experts"].tolist(), [[1, 1, 2]])

    def test_pcgrad_only_projects_conflicting_gradient(self):
        import torch

        from morphoroute.models.pcgrad import project_conflicting_pair

        first = torch.tensor([1.0, 0.0])
        aligned = torch.tensor([2.0, 0.0])
        conflicting = torch.tensor([-1.0, 1.0])
        unchanged, _ = project_conflicting_pair(first, aligned)
        projected, _ = project_conflicting_pair(first, conflicting)
        torch.testing.assert_close(unchanged, first)
        self.assertGreaterEqual(torch.dot(projected, conflicting).item(), -1e-6)


if __name__ == "__main__":
    unittest.main()
