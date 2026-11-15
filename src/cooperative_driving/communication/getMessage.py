import math


MAX_TIME_TO_COLLISION = 10.0


def _forward_speed(vehicle):
    """Return the vehicle speed along the road's forward direction."""
    return float(vehicle.speed * math.cos(vehicle.heading))


def _time_to_collision(vehicle):
    """Estimate TTC to the closest slower vehicle in the same lane.

    A capped value means that no immediate collision is predicted. This uses
    simulator state, so it does not require adding a radar sensor.
    """
    if vehicle.road is None or vehicle.lane_index is None:
        return MAX_TIME_TO_COLLISION

    vehicles_ahead = [
        other
        for other in vehicle.road.vehicles
        if other is not vehicle
        and other.lane_index == vehicle.lane_index
        and other.position[0] > vehicle.position[0]
    ]
    if not vehicles_ahead:
        return MAX_TIME_TO_COLLISION

    front_vehicle = min(
        vehicles_ahead,
        key=lambda other: other.position[0] - vehicle.position[0],
    )
    gap = float(front_vehicle.position[0] - vehicle.position[0])
    closing_speed = _forward_speed(vehicle) - _forward_speed(front_vehicle)
    if gap <= 0 or closing_speed <= 0:
        return MAX_TIME_TO_COLLISION

    return min(MAX_TIME_TO_COLLISION, gap / closing_speed)


DRIVER_TYPE_CODES = {"yielding": 1.0, "aggressive": -1.0}
MESSAGE_FEATURES = ("driver_type", "intent_acceleration", "target_lane", "time_to_collision")


def getMessage(car):
    """Return private intent and safety information shared by a vehicle."""
    target_lane = (
        car.target_lane_index[2]
        if hasattr(car, "target_lane_index")
        else car.lane_index[2]
    )
    return {
        "vehicle_id": car.vehicle_id,
        # Private driver type: only revealed when this car is asked.
        "driver_type": getattr(car, "driver_type", None),
        "intent": car.intent.copy(),
        "target_lane": int(target_lane),
        "time_to_collision": _time_to_collision(car),
    }


def encode_message(car):
    """Return the 4-value message vector used by MLPx and the datasets.

    Order: driver type (+1 yielding, -1 aggressive, 0 unknown/other),
    intent acceleration, target lane, time to collision.
    """
    message = getMessage(car)
    return [
        DRIVER_TYPE_CODES.get(message["driver_type"], 0.0),
        float(message["intent"]["intent_acceleration"]),
        float(message["target_lane"]),
        float(message["time_to_collision"]),
    ]
