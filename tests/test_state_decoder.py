import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from scripts.training.train_state_decoder import loss_fn, split_by_episode
from src.cooperative_driving.models.decoder import ARCHITECTURES, StateDecoder, load_decoder
from src.cooperative_driving.visualization.frames import (
    FRAME_HEIGHT, FRAME_WIDTH, ROAD_GREY, iou, lane_line_rows, make_frame_environment, near_vehicles,
    pixel_offsets_m, to_gray, vehicle_mask,
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
        target = torch.full((1, 1, 4, 4), 100, dtype=torch.uint8)
        target[..., 0, 0] = 160  # one vehicle pixel
        miss_vehicle = torch.zeros(1, 1, 4, 4)
        miss_vehicle[..., 0, 0] = -5.0
        miss_road = torch.zeros(1, 1, 4, 4)
        miss_road[..., 1, 1] = 5.0
        self.assertGreater(loss_fn(miss_vehicle, target, 20.0), loss_fn(miss_road, target, 20.0))

    def test_loss_weights_use_exact_gray_thresholds(self):
        # Gray 120 is exactly at the threshold: not a vehicle pixel, weight 1.
        target = torch.full((1, 1, 2, 2), 120, dtype=torch.uint8)
        logits = torch.zeros(1, 1, 2, 2)
        self.assertAlmostEqual(float(loss_fn(logits, target, 20.0)), float(loss_fn(logits, target, 0.0)))

    def test_split_keeps_each_episode_in_one_group(self):
        episodes = np.repeat(np.arange(20), 3)
        train, validation, test = split_by_episode(episodes)
        groups = [set(episodes[indices]) for indices in (train, validation, test)]
        self.assertFalse(groups[0] & groups[1] or groups[0] & groups[2] or groups[1] & groups[2])
        self.assertEqual(sorted(groups[2]), [0, 5, 10, 15])
        self.assertEqual(sorted(groups[1]), [1, 6, 11, 16])
        self.assertEqual(len(train) + len(validation) + len(test), len(episodes))

    def test_geometry_is_not_saved_in_checkpoints(self):
        keys = StateDecoder().state_dict().keys()
        self.assertFalse({"pixels_m", "car_size", "state_scale"} & set(keys))


class CameraGeometryTests(unittest.TestCase):
    def test_pixel_offsets_match_the_viewer(self):
        environment = make_frame_environment()
        environment.reset(seed=1)
        environment.render()
        surface = environment.viewer.sim_surface
        offsets = pixel_offsets_m(FRAME_HEIGHT, FRAME_WIDTH).reshape(FRAME_HEIGHT, FRAME_WIDTH, 2)
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

    def test_mask_drops_a_lane_stripe_but_keeps_a_car_on_a_lane_line(self):
        state = np.zeros((11, 5), dtype=np.float32)
        state[0] = [1, 0.0, 6.0 / 16, 0.6, 0]      # ego at road y = 6 m, on a lane line
        state[1] = [1, 30.0 / 200, 0.0, 0.0, 0]     # a car 30 m ahead, same y
        gray = np.full((FRAME_HEIGHT, FRAME_WIDTH), ROAD_GREY, dtype=np.uint8)
        line_row = FRAME_HEIGHT // 2                # the ego row = the line row
        gray[line_row, :] = 160                     # a soft grey stripe along the whole line
        mask = vehicle_mask(gray, state.reshape(-1))
        near = near_vehicles(state.reshape(-1), FRAME_HEIGHT, FRAME_WIDTH)
        self.assertTrue(mask[line_row][near[line_row]].all())      # car pixels on the line stay
        self.assertFalse(mask[line_row][~near[line_row]].any())    # the stripe elsewhere goes
        self.assertTrue(vehicle_mask(gray)[line_row].all())         # without a state, all count


if __name__ == "__main__":
    unittest.main()
