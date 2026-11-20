import unittest

import numpy as np

from src.cooperative_driving.environments.customEnv import CustomEnv
from src.cooperative_driving.visualization import StateRenderer, decode_state


class StateRendererTests(unittest.TestCase):
    def test_decoded_positions_match_the_simulator(self):
        environment = CustomEnv()
        observation, _ = environment.reset(seed=3)
        for _ in range(3):
            decoded = decode_state(observation)
            true_positions = np.array([vehicle.position for vehicle in environment.road.vehicles])
            # The ego stays within 200 m of the start here, so its x is not clipped.
            np.testing.assert_allclose(
                [decoded[0]["x"], decoded[0]["y"]], environment.vehicle.position, atol=1e-3
            )
            for item in decoded:
                distances = np.linalg.norm(true_positions - [item["x"], item["y"]], axis=1)
                self.assertLess(distances.min(), 1e-3)
            observation, *_ = environment.step(1)  # IDLE
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
