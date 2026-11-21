import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_state_decoder import loss_fn, split_by_episode
from src.cooperative_driving.models.decoder import ARCHITECTURES, StateDecoder, load_decoder
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_WIDTH, iou, lane_line_rows, make_frame_environment, to_gray, vehicle_mask,
)


class StateDecoderTests(unittest.TestCase):
    def test_output_shape_matches_the_frame_size(self):
        for name, model_class in ARCHITECTURES.items():
            with self.subTest(architecture=name):
                model = model_class()
                logits = model(torch.zeros(2, StateDecoder.STATE_SIZE))
                self.assertEqual(tuple(logits.shape), (2, 1, FRAME_HEIGHT, FRAME_WIDTH))
                frame = model.draw(np.zeros(StateDecoder.STATE_SIZE))
                self.assertEqual(frame.shape, (FRAME_HEIGHT, FRAME_WIDTH))
                self.assertEqual(frame.dtype, np.uint8)

    def test_save_and_load_give_the_same_output(self):
        state = np.random.default_rng(0).uniform(-1, 1, StateDecoder.STATE_SIZE)
        for name, model_class in ARCHITECTURES.items():
            with self.subTest(architecture=name), tempfile.TemporaryDirectory() as folder:
                model = model_class().eval()
                path = Path(folder) / "decoder.pt"
                torch.save({"model_state_dict": model.state_dict(), "model_kwargs": {},
                            "architecture": name}, path)
                loaded = load_decoder(path)
                self.assertIsInstance(loaded, model_class)
                np.testing.assert_array_equal(loaded.draw(state), model.draw(state))

    def test_vehicle_decoder_ignores_vehicle_order(self):
        model = StateDecoder().eval()
        rows = np.zeros((11, 5), dtype=np.float32)
        rows[:3] = [[1, 0.1, 0.25, 0.3, 0], [1, 0.05, 0.25, 0, 0], [1, -0.08, -0.25, 0.02, 0]]
        swapped = rows.copy()
        swapped[[1, 2]] = rows[[2, 1]]
        with torch.no_grad():
            first = model(torch.as_tensor(rows.reshape(1, -1)))
            second = model(torch.as_tensor(swapped.reshape(1, -1)))
        torch.testing.assert_close(first, second)

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


class CameraGeometryTests(unittest.TestCase):
    def test_pixel_offsets_match_the_viewer(self):
        from src.cooperative_driving.models.decoder.state_decoder import pixel_offsets_m

        environment = make_frame_environment()
        environment.reset(seed=1)
        environment.render()
        surface = environment.viewer.sim_surface
        offsets = pixel_offsets_m(FRAME_HEIGHT, FRAME_WIDTH).reshape(FRAME_HEIGHT, FRAME_WIDTH, 2).numpy()
        ego = environment.vehicle.position
        for vehicle in environment.road.vehicles:
            column, row = surface.vec2pix(vehicle.position)
            if 0 <= row < FRAME_HEIGHT and 0 <= column < FRAME_WIDTH:
                # Within one pixel (about 0.43 m) of the true offset.
                np.testing.assert_allclose(offsets[row, column], vehicle.position - ego, atol=0.5)
        environment.close()


class FrameTests(unittest.TestCase):
    def test_frame_render_has_vehicles(self):
        environment = make_frame_environment()
        environment.reset(seed=0)
        gray = to_gray(environment.render())
        self.assertEqual(gray.shape, (FRAME_HEIGHT, FRAME_WIDTH))
        self.assertGreater(vehicle_mask(gray).sum(), 0)
        self.assertEqual(iou(vehicle_mask(gray), vehicle_mask(gray)), 1.0)
        environment.close()

    def test_lane_line_rows_cover_the_white_lines(self):
        environment = make_frame_environment()
        for seed in range(5):
            observation, _ = environment.reset(seed=seed)
            gray = to_gray(environment.render())
            line_rows = lane_line_rows(observation[0][2])
            white_rows = np.flatnonzero((gray > 230).any(axis=1))
            self.assertGreater(len(white_rows), 0)
            self.assertTrue(line_rows[white_rows].all(), f"seed {seed}: {white_rows}")
        environment.close()


if __name__ == "__main__":
    unittest.main()
