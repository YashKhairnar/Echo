from highway_env.vehicle.behavior import IDMVehicle
from highway_env.vehicle.controller import ControlledVehicle

class IntentMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.intent = {
            # Positive = accelerate, negative = decelerate/brake, zero = maintain speed.
            "intent_acceleration": 0.0,
            # Low-level steering intent. Zero means no steering input. Positive is steer left and negative is steer right
            "intent_steering": 0.0,
            # Desired lane index minus current lane index. Zero means stay in lane.
            "intent_lane_change": 0.0,
        }

    def record_intent(self):
        self.intent.update({
            # A negative acceleration represents deceleration.
            "intent_acceleration": self.action.get("acceleration", 0.0),
            # The vehicle's current steering command.
            "intent_steering": self.action.get("steering", 0.0),
            # Positive/negative values indicate a target lane change direction.
            "intent_lane_change": (
                self.target_lane_index[2] - self.lane_index[2]
                if hasattr(self, "target_lane_index")
                else 0
            ),
        })

    def act(self, action=None):
        result = super().act(action)
        self.record_intent()
        return result

    def to_dict(self, origin_vehicle=None, observe_intentions=True):
        data = super().to_dict(origin_vehicle, observe_intentions)
        data.update(self.intent)
        data["vehicle_id"] = self.vehicle_id
        return data


class IntentIDMVehicle(IntentMixin, IDMVehicle):
    pass


class IntentControlledVehicle(IntentMixin, ControlledVehicle):
    pass

