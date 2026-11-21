import unittest
from types import SimpleNamespace

import numpy as np

from src.cooperative_driving.planner.safety import guard_action


def vehicle(x=0, speed=30, lane=1, target=None, vehicle_id=0):
    return SimpleNamespace(position=np.array([float(x), lane * 4.]),
                           velocity=np.array([float(speed), 0.]),
                           lane_index=("0", "1", lane),
                           target_lane_index=("0", "1", lane if target is None else target),
                           LENGTH=5., vehicle_id=vehicle_id)


class GuardTests(unittest.TestCase):
    def test_preserves_safe_action(self):
        action, info = guard_action([0, 1, 2, 3, 0], vehicle(), {}, 4)
        self.assertEqual(action, 3)
        self.assertFalse(info["overridden"])
        self.assertIsNone(info["ttc"])

    def test_brakes_for_ttc_using_bumper_gap(self):
        ego, front = vehicle(), vehicle(x=25, speed=20, vehicle_id=7)
        action, info = guard_action([0, 3, 0, 2, 1], ego, {"front": {"vehicle": front}}, 4)
        self.assertEqual(action, 4)
        self.assertEqual(info["front_clearance"], 20)
        self.assertEqual(info["ttc"], 2)
        self.assertEqual(info["front_vehicle_id"], 7)
        self.assertTrue(info["overridden"])

    def test_small_gap_brakes_even_without_closing(self):
        action, info = guard_action([0, 3, 0, 2, 1], vehicle(speed=20),
                                  {"front": {"vehicle": vehicle(x=10, speed=25)}}, 4)
        self.assertEqual(action, 4)
        self.assertIsNone(info["ttc"])
        self.assertIn("front_clearance", info["reasons"])

    def test_ignores_rear_and_adjacent_vehicles(self):
        visible = {"rear": {"vehicle": vehicle(x=-2)},
                   "adjacent": {"vehicle": vehicle(x=6, lane=2)}}
        action, info = guard_action([0, 3, 0, 2, 1], vehicle(), visible, 4)
        self.assertEqual(action, 1)
        self.assertFalse(info["braking_required"])

    def test_masks_current_and_target_road_boundaries(self):
        for ego, scores, invalid in [(vehicle(lane=0), [10, 2, 1, 0, -1], 0),
                                     (vehicle(lane=3), [0, 2, 10, 1, -1], 2),
                                     (vehicle(lane=2, target=3), [0, 2, 10, 1, -1], 2)]:
            action, info = guard_action(scores, ego, {}, 4)
            self.assertNotEqual(action, invalid)
            self.assertNotIn(invalid, info["allowed_actions"])
            self.assertIn("boundary_lane_mask", info["reasons"])

    def test_braking_trigger_is_logged_when_already_slower(self):
        action, info = guard_action([0, 0, 0, 0, 2], vehicle(),
                                  {"front": {"vehicle": vehicle(x=6)}}, 4)
        self.assertEqual(action, 4)
        self.assertTrue(info["braking_required"])
        self.assertFalse(info["overridden"])

    def test_destination_lane_blocks_beside_front_and_closing_rear(self):
        for neighbor in [vehicle(x=1, lane=2), vehicle(x=12, lane=2),
                         vehicle(x=-25, speed=40, lane=2)]:
            action, info = guard_action([0, 2, 5, 1, 0], vehicle(),
                                       {'candidate': {'vehicle': neighbor}}, 4,
                                       check_lane_changes=True)
            self.assertEqual(action, 1)
            self.assertNotIn(2, info['allowed_actions'])
            self.assertIn('destination_lane_gap', info['reasons'])

    def test_safe_lane_changes_preserved_in_both_directions(self):
        for command, destination in [(0, 0), (2, 2)]:
            values = [0, 2, 0, 1, 0]
            values[command] = 5
            action, info = guard_action(values, vehicle(),
                                       {'far': {'vehicle': vehicle(x=40, speed=30, lane=destination)}},
                                       4, check_lane_changes=True)
            self.assertEqual(action, command)
            self.assertFalse(info['overridden'])

    def test_blocks_stacked_commands_during_lane_change(self):
        action, info = guard_action([5, 2, 3, 1, 0], vehicle(lane=1, target=2), {},
                                   4, check_lane_changes=True)
        self.assertEqual(action, 1)
        self.assertIn('maneuver_in_progress', info['reasons'])

    def test_destination_check_is_opt_in(self):
        action, _ = guard_action([0, 2, 5, 1, 0], vehicle(),
                                {'beside': {'vehicle': vehicle(x=0, lane=2)}}, 4)
        self.assertEqual(action, 2)

    def test_closing_front_blocked_before_current_gap_is_small(self):
        action, info = guard_action([0, 2, 5, 1, 0], vehicle(speed=30),
                                   {'front': {'vehicle': vehicle(x=40, speed=20, lane=2)}},
                                   4, check_lane_changes=True)
        self.assertEqual(action, 1)
        self.assertEqual(info['lane_change_checks']['2']['conflicts'][0]['projected_clearance'], 5.)

    def test_receding_rear_with_clear_gap_does_not_block(self):
        action, _ = guard_action([0, 2, 5, 1, 0], vehicle(speed=30),
                                {'rear': {'vehicle': vehicle(x=-20, speed=20, lane=2)}},
                                4, check_lane_changes=True)
        self.assertEqual(action, 2)

    def test_lateral_motion_blocks_command_even_when_lane_indices_match(self):
        ego = vehicle()
        ego.heading = .1
        action, info = guard_action([5, 2, 3, 1, 0], ego, {}, 4,
                                   check_lane_changes=True)
        self.assertEqual(action, 1)
        self.assertIn('maneuver_in_progress', info['reasons'])

    def test_speed_cap_masks_acceleration_without_mutating_scores(self):
        ego = vehicle()
        ego.target_speed = 30.
        values = [0., 2., 1., 5., 0.]
        action, info = guard_action(values, ego, {}, 4, max_target_speed=30.)
        self.assertEqual(action, 1)
        self.assertIn('target_speed_cap', info['reasons'])
        self.assertEqual(info['raw_action'], 3)
        self.assertNotIn(3, info['allowed_actions'])
        self.assertEqual(values[3], 5.)
        ego.target_speed = 25.
        action, _ = guard_action(values, ego, {}, 4, max_target_speed=30.)
        self.assertEqual(action, 3)

    def test_speed_cap_retains_emergency_braking(self):
        ego = vehicle()
        ego.target_speed = 30.
        action, info = guard_action([0, 2, 1, 5, 0], ego,
                                   {'front': {'vehicle': vehicle(x=6, speed=20)}},
                                   4, max_target_speed=30., check_lane_changes=True)
        self.assertEqual(action, 4)
        self.assertTrue(info['braking_required'])

    def test_comfort_window_prefers_a_safe_lane_change(self):
        ego = vehicle(speed=30)
        front = vehicle(x=45, speed=20, lane=1)
        action, info = guard_action(
            [4, 1, 3, 5, 0], ego, {'front': {'vehicle': front}}, 4,
            check_lane_changes=True, comfort_ttc=6.,
        )
        self.assertEqual(action, 0)
        self.assertIn('front_caution', info['reasons'])
        self.assertIn('proactive_lane_change', info['reasons'])
        self.assertNotIn(3, info['allowed_actions'])

    def test_comfort_window_uses_other_lane_when_best_lane_is_blocked(self):
        ego = vehicle(speed=30)
        visible = {
            'front': {'vehicle': vehicle(x=45, speed=20, lane=1)},
            'left_beside': {'vehicle': vehicle(x=0, speed=30, lane=0)},
        }
        action, info = guard_action(
            [5, 1, 3, 4, 0], ego, visible, 4,
            check_lane_changes=True, comfort_ttc=6.,
        )
        self.assertEqual(action, 2)
        self.assertNotIn(0, info['allowed_actions'])

    def test_comfort_lane_change_records_forward_clearance_gain(self):
        ego = vehicle(speed=30)
        visible = {
            'front': {'vehicle': vehicle(x=45, speed=20, lane=1)},
            'left_front': {'vehicle': vehicle(x=50, speed=30, lane=0)},
        }
        action, info = guard_action(
            [5, 1, 3, 4, 0], ego, visible, 4,
            check_lane_changes=True, comfort_ttc=6.,
        )
        self.assertEqual(action, 0)
        self.assertEqual(info['lane_change_checks']['0']['clearance_gain'], 5.)
        self.assertEqual(info['lane_change_checks']['2']['front_clearance'], float('inf'))

    def test_lane_change_cooldown_prevents_ping_pong(self):
        state = {}
        ego = vehicle(speed=30)
        visible = {'front': {'vehicle': vehicle(x=45, speed=20, lane=1)}}
        action, info = guard_action(
            [4, 1, 3, 5, 0], ego, visible, 4,
            check_lane_changes=True, comfort_ttc=6., controller_state=state,
        )
        self.assertEqual(action, 0)
        self.assertEqual(state['lane_change_cooldown'], 5)
        action, info = guard_action(
            [3, 1, 5, 4, 0], ego, visible, 4,
            check_lane_changes=True, comfort_ttc=6., controller_state=state,
        )
        self.assertNotIn(action, (0, 2))
        self.assertEqual(info['lane_change_cooldown_remaining'], 5)
        self.assertEqual(state['lane_change_cooldown'], 4)

    def test_comfort_window_slows_when_both_lanes_are_blocked(self):
        ego = vehicle(speed=30)
        visible = {
            'front': {'vehicle': vehicle(x=45, speed=20, lane=1)},
            'left_beside': {'vehicle': vehicle(x=0, speed=30, lane=0)},
            'right_beside': {'vehicle': vehicle(x=0, speed=30, lane=2)},
        }
        action, info = guard_action(
            [5, 1, 3, 4, 0], ego, visible, 4,
            check_lane_changes=True, comfort_ttc=6.,
        )
        self.assertEqual(action, 4)
        self.assertIn('proactive_slowdown', info['reasons'])

    def test_braking_creates_acceleration_cooldown(self):
        state = {}
        ego = vehicle(speed=30)
        action, info = guard_action(
            [0, 1, 2, 3, 5], ego, {}, 4,
            check_lane_changes=True, comfort_ttc=6., controller_state=state,
        )
        self.assertEqual(action, 4)
        self.assertEqual(state['acceleration_cooldown'], 3)
        action, info = guard_action(
            [0, 1, 2, 5, 0], ego, {}, 4,
            check_lane_changes=True, comfort_ttc=6., controller_state=state,
        )
        self.assertNotEqual(action, 3)
        self.assertIn('acceleration_cooldown', info['reasons'])
        self.assertEqual(state['acceleration_cooldown'], 2)

    def test_comfort_window_keeps_emergency_braking_priority(self):
        ego = vehicle(speed=30)
        action, info = guard_action(
            [5, 1, 3, 4, 0], ego,
            {'front': {'vehicle': vehicle(x=25, speed=20, lane=1)}}, 4,
            check_lane_changes=True, comfort_ttc=6.,
        )
        self.assertEqual(action, 4)
        self.assertTrue(info['braking_required'])

    def test_comfort_window_requires_checked_lane_changes(self):
        with self.assertRaises(ValueError):
            guard_action([0, 1, 2, 3, 4], vehicle(), {}, 4, comfort_ttc=6.)


if __name__ == "__main__":
    unittest.main()
