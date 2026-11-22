import random
import unittest

import numpy as np
import torch

from scripts.evaluation.visualize_mlpx_rollout import rollout, scene_rollout, summarize, visible_messages
from src.cooperative_driving.environments.intent_scenarios import intent_episode, matched_intent_pair
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

    def test_challenge_scene_modes_start_from_the_same_state_and_idle(self):
        model = MLPxWorldModel(state_size=57).eval()
        results = {}
        for mode in ("stay", "merge"):
            results[mode] = scene_rollout(400_000, mode, model, self.renderer, self.decoder,
                                          horizon=2, messages="all")
            frames, ious, mses = results[mode]
            self.assertGreater(len(frames), 0)
            self.assertEqual(frames[0][0].shape, (FRAME_HEIGHT, FRAME_WIDTH))
        # The first real frames differ only by what the neighbour does in step 1,
        # but both modes start from one public observation (checked in the env).
        stay, merge = intent_episode(400_000, "stay"), intent_episode(400_000, "merge")
        np.testing.assert_array_equal(stay.observation_type.observe(), merge.observation_type.observe())
        stay.close()
        merge.close()

    def test_neighbour_messages_fill_only_the_neighbour_slot(self):
        environments, metadata = matched_intent_pair(400_000, duration=2)
        try:
            environment = environments["merge"]
            everyone = visible_messages(environment)["messages_by_slot"]
            neighbour = visible_messages(environment, metadata["neighbor_id"])["messages_by_slot"]
            self.assertGreater(sum(item is not None for item in everyone), 1)
            filled = [item for item in neighbour if item is not None]
            self.assertEqual(len(filled), 1)
            self.assertEqual(filled[0]["vehicle_id"], metadata["neighbor_id"])
        finally:
            for environment in environments.values():
                environment.close()

    def test_summary_counts_only_complete_rollouts(self):
        frame = np.zeros((FRAME_HEIGHT, FRAME_WIDTH), dtype=np.uint8)
        complete = ([(frame, frame, frame)] * 2, [0.5, 0.25], [0.1, 0.2])
        short = ([(frame, frame, frame)], [1.0], [0.0])
        summary = summarize([complete, short, None], horizon=2)
        self.assertEqual(summary["complete_rollouts"], 1)
        self.assertEqual(summary["skipped_rollouts"], 2)
        self.assertEqual([step["drawn_vehicle_iou"] for step in summary["per_step"]], [0.5, 0.25])


if __name__ == "__main__":
    unittest.main()
