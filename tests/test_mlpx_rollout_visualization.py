import random
import unittest

import torch

from scripts.evaluation.visualize_mlpx_rollout import rollout
from src.cooperative_driving.models.decoder import StateDecoder
from src.cooperative_driving.models.MLPx.mlpx import MLPxWorldModel
from src.cooperative_driving.visualization import StateRenderer
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_SCALING, FRAME_WIDTH, make_frame_environment,
)


class RolloutVisualizationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        cls.environment = make_frame_environment()
        cls.renderer = StateRenderer(FRAME_WIDTH, FRAME_HEIGHT, FRAME_SCALING)
        cls.decoder = StateDecoder().eval()

    @classmethod
    def tearDownClass(cls):
        cls.renderer.close()
        cls.environment.close()

    def test_rollout_works_with_55_and_57_value_models(self):
        for state_size in (55, 57):
            for messages in ("none", "all"):
                with self.subTest(state_size=state_size, messages=messages):
                    model = MLPxWorldModel(state_size=state_size).eval()
                    result = rollout(self.environment, model, self.renderer, self.decoder,
                                     seed=0, warmup=0, horizon=2, messages=messages,
                                     rng=random.Random(0))
                    self.assertIsNotNone(result)
                    frames, ious, mses = result
                    self.assertGreater(len(frames), 0)
                    self.assertEqual(len(frames), len(ious))
                    self.assertEqual(len(frames), len(mses))
                    for real, drawn, decoded in frames:
                        for image in (real, drawn, decoded):
                            self.assertEqual(image.shape, (FRAME_HEIGHT, FRAME_WIDTH))
                    self.assertTrue(all(0.0 <= score <= 1.0 for score in ious))


if __name__ == "__main__":
    unittest.main()
