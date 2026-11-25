import unittest
from types import SimpleNamespace

import numpy as np

from src.cooperative_driving.planner.safety import merge_brake_action, merging_neighbours


def car(vehicle_id, lane, x, speed):
    return SimpleNamespace(vehicle_id=vehicle_id, lane_index=('0', '1', lane),
                           position=np.array([x, 0.0]), velocity=np.array([speed, 0.0]),
                           LENGTH=5.0)


class FakeEnvironment:
    def __init__(self, ego, others):
        self.vehicle = ego
        self.others = {vehicle.vehicle_id: vehicle for vehicle in others}

    def get_vehicle(self, vehicle_id):
        return self.others[vehicle_id]


def message(vehicle_id, target_lane):
    return {'messages_by_slot': [None, {'vehicle_id': vehicle_id, 'crashed': False,
                                        'target_lane': ('0', '1', target_lane)}]}


class MergeBrakeTests(unittest.TestCase):
    def setUp(self):
        self.ego = car(0, 1, 100.0, 28.0)

    def test_close_merging_car_forces_brake(self):
        environment = FakeEnvironment(self.ego, [car(7, 2, 110.0, 25.0)])
        action, report = merge_brake_action(3, environment, message(7, 1))
        self.assertEqual(action, 4)
        self.assertTrue(report['changed'])

    def test_far_merging_car_blocks_acceleration_only(self):
        environment = FakeEnvironment(self.ego, [car(7, 2, 200.0, 30.0)])
        self.assertEqual(merge_brake_action(3, environment, message(7, 1))[0], 1)
        self.assertEqual(merge_brake_action(0, environment, message(7, 1))[0], 0)

    def test_car_staying_in_its_lane_changes_nothing(self):
        environment = FakeEnvironment(self.ego, [car(7, 2, 110.0, 25.0)])
        self.assertEqual(merge_brake_action(3, environment, message(7, 2)), (3, None))

    def test_car_fully_behind_is_ignored(self):
        environment = FakeEnvironment(self.ego, [car(7, 2, 80.0, 35.0)])
        self.assertEqual(merging_neighbours(environment, message(7, 1)), [])

    def test_car_already_in_ego_lane_is_left_to_the_guard(self):
        environment = FakeEnvironment(self.ego, [car(7, 1, 110.0, 25.0)])
        self.assertEqual(merging_neighbours(environment, message(7, 1)), [])

    def test_no_message_changes_nothing(self):
        environment = FakeEnvironment(self.ego, [])
        self.assertEqual(merge_brake_action(3, environment, None), (3, None))


if __name__ == '__main__':
    unittest.main()
