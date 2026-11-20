import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_state_decoder import loss_fn, split_by_episode
from src.cooperative_driving.models.decoder import StateDecoder
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_WIDTH, iou, make_frame_environment, to_gray, vehicle_mask,
)


class StateDecoderTests(unittest.TestCase):
    def test_output_shape_matches_the_frame_size(self):
        model = StateDecoder()
        logits = model(torch.zeros(2, StateDecoder.STATE_SIZE))
        self.assertEqual(tuple(logits.shape), (2, 1, FRAME_HEIGHT, FRAME_WIDTH))
        frame = model.draw(np.zeros(StateDecoder.STATE_SIZE))
        self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH))
        self.assertEqual(frame.dtype, np.uint8)

    def test_save_and_load_give_the_same_output(self):
        model = StateDecoder()
        state = np.random.default_rng(0).uniform(-1, 1, StateDecoder.STATE_SIZE)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "decoder.pt"
            torch.save({"model_state_dict": model.state_dict(), "model_kwargs": {}}, path)
            np.testing.assert_array_equal(StateDecoder.load(path).draw(state), model.eval().draw(state))

    def test_vehicle_pixels_weigh_more_in_the_loss(self):
        target = torch.full((1, 1, 4, 4), 100 / 255)
        target[..., 0, 0] = 160 / 255  # one vehicle pixel
        miss_vehicle = torch.zeros_like(target)
        miss_vehicle[..., 0, 0] = -5.0
        miss_road = torch.zeros_like(target)
        miss_road[..., 1, 1] = 5.0
        self.assertGreater(loss_fn(miss_vehicle, target, 20.0), loss_fn(miss_road, target, 20.0))

    def test_split_keeps_each_episode_on_one_side(self):
        episodes = np.repeat(np.arange(20), 3)
        train, test = split_by_episode(episodes)
        self.assertFalse(set(episodes[train]) & set(episodes[test]))
        self.assertEqual(sorted(set(episodes[test])), [0, 5, 10, 15])


class FrameTests(unittest.TestCase):
    def test_frame_render_has_vehicles(self):
        environment = make_frame_environment()
        environment.reset(seed=0)
        gray = to_gray(environment.render())
        self.assertEqual(gray.shape, (FRAME_HEIGHT, FRAME_WIDTH))
        self.assertGreater(vehicle_mask(gray).sum(), 0)
        self.assertEqual(iou(vehicle_mask(gray), vehicle_mask(gray)), 1.0)
        environment.close()


if __name__ == "__main__":
    unittest.main()
