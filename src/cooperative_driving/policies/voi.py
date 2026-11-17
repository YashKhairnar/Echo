"""Value-of-information request rule: ask a car only if its answer could change the plan.

No learning: it uses the trained MLPx planner to imagine both possible
answers of every car in the target lane.

For a car k, with P(yielding) = 0.5:

    know_answer  = 0.5 * best value if k says "yielding"
                 + 0.5 * best value if k says "aggressive"
    dont_know    = best value of one action chosen without the answer
                   (the action with the best 0.5/0.5 average)
    value of asking k = know_answer - dont_know

It is 0 when the same move is best whatever the car answers (after the
merge, or for a car whose type doesn't matter, like the one in front), and
positive when the answer would change the move. The rule asks the car with
the highest value, if that value is larger than the cost of a question.

A message has four values, but only the driver type is imagined both ways;
the other three are filled with what platoon cars almost always say
(acceleration 0, target lane 0, time to collision 10).
"""

import torch

from ..communication.request_utils import (
    MAX_REQUEST_SLOTS,
    MESSAGE_FEATURES,
    vehicle_by_id,
    visible_neighbor_ids,
)
from ..environments.customEnv import TARGET_LANE

YIELDING_CODE, AGGRESSIVE_CODE = 1.0, -1.0
TYPICAL_REST_OF_MESSAGE = [0.0, 0.0, 10.0]  # acceleration, target lane, time to collision


def imagined_message(slot, driver_type_code):
    request_mask = [0] * MAX_REQUEST_SLOTS
    messages = [[0.0] * MESSAGE_FEATURES for _ in range(MAX_REQUEST_SLOTS)]
    request_mask[slot] = 1
    messages[slot] = [driver_type_code] + TYPICAL_REST_OF_MESSAGE
    return {"request_mask": request_mask, "messages": messages}


class AskValueOfInformation:
    name = "voi"

    def __init__(self, planner, cost, yielding_probability=0.5):
        self.planner = planner
        self.cost = cost
        self.p = yielding_probability

    def reset(self):
        pass

    def value_of_asking(self, slot):
        """Value of learning the driver type of the car in ``slot``."""
        if_yielding = torch.tensor(self.planner.action_values_batched(
            self.observation, imagined_message(slot, YIELDING_CODE)))
        if_aggressive = torch.tensor(self.planner.action_values_batched(
            self.observation, imagined_message(slot, AGGRESSIVE_CODE)))
        know_answer = self.p * if_yielding.max() + (1 - self.p) * if_aggressive.max()
        dont_know = (self.p * if_yielding + (1 - self.p) * if_aggressive).max()
        return float(know_answer - dont_know)

    def target_lane_slots(self, environment, observation):
        return [
            slot for slot, vehicle_id in enumerate(visible_neighbor_ids(observation))
            if (vehicle := vehicle_by_id(environment, vehicle_id)) is not None
            and vehicle.lane_index[2] == TARGET_LANE
        ]

    def slots(self, environment, observation):
        self.observation = observation
        values = {slot: self.value_of_asking(slot) for slot in self.target_lane_slots(environment, observation)}
        if not values:
            return []
        best = max(values, key=values.get)
        return [best] if values[best] > self.cost else []
