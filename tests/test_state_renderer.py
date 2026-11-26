import unittest

import numpy as np

from scripts.evaluation.state_drawing_metrics import observed_vehicles, position_errors
from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.visualization import StateRenderer, decode_state


class StateRendererTests(unittest.TestCase):
    def test_decoded_positions_match_the_simulator(self):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=3)
        for _ in range(3):
            decoded = decode_state(observation)
            # The ego stays within 200 m of the start here, so its x is not clipped.
            np.testing.assert_allclose(
                [decoded[0]["x"], decoded[0]["y"]], environment.vehicle.position, atol=1e-3
            )
            # Each row is compared with the exact car that the observation put there.
            vehicles = observed_vehicles(environment)
            self.assertEqual(len(vehicles), len(decoded) - 1)
            for item, vehicle in zip(decoded[1:], vehicles):
                np.testing.assert_allclose([item["x"], item["y"]], vehicle.position, atol=1e-3)
            observation, *_ = environment.step(1)  # IDLE
        environment.close()

    def test_position_errors_use_the_observed_order(self):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=4)
        errors, ego_y_error = position_errors(observation, environment)
        self.assertGreater(len(errors), 0)
        self.assertLess(max(errors), 1e-3)
        self.assertLess(ego_y_error, 1e-3)
        environment.close()

    def test_draw_returns_an_rgb_image(self):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=0)
        renderer = StateRenderer(screen_width=300, screen_height=80, scaling=4.0)
        image = renderer.draw(observation)
        self.assertEqual(image.shape, (80, 300, 3))
        self.assertEqual(image.dtype, np.uint8)
        self.assertGreater(len(np.unique(image.reshape(-1, 3), axis=0)), 2)
        renderer.close()
        environment.close()


if __name__ == "__main__":
    unittest.main()
