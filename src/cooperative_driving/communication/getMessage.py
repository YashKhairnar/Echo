"""What a car says when it is asked.

A message has four values:
    driver_type          +1 yielding, -1 aggressive, 0 other (hidden otherwise!)
    intent_acceleration  the car's current acceleration [m/s2]
    target_lane          the lane the car is heading for
    time_to_collision    seconds until it would hit the car ahead (max 10)
"""

import math

MAX_TIME_TO_COLLISION = 10.0
DRIVER_TYPE_CODES = {"yielding": 1.0, "aggressive": -1.0}


def getMessage(car):
    """Return the message of ``car`` as a readable dictionary."""
    target_lane = car.target_lane_index[2] if hasattr(car, "target_lane_index") else car.lane_index[2]
    return {
        "vehicle_id": car.vehicle_id,
        "driver_type": getattr(car, "driver_type", None),
        "intent_acceleration": float(car.intent["intent_acceleration"]),
        "target_lane": int(target_lane),
        "time_to_collision": time_to_collision(car),
    }


def encode_message(car):
    """Return the message of ``car`` as the 4 numbers MLPx reads."""
    message = getMessage(car)
    return [
        DRIVER_TYPE_CODES.get(message["driver_type"], 0.0),
        message["intent_acceleration"],
        float(message["target_lane"]),
        float(message["time_to_collision"]),
    ]


def time_to_collision(car):
    """Seconds until ``car`` reaches the closest car ahead in its lane (capped)."""
    if car.road is None or car.lane_index is None:
        return MAX_TIME_TO_COLLISION
    ahead = [
        other for other in car.road.vehicles
        if other is not car
        and other.lane_index == car.lane_index
        and other.position[0] > car.position[0]
    ]
    if not ahead:
        return MAX_TIME_TO_COLLISION
    front = min(ahead, key=lambda other: other.position[0])
    gap = float(front.position[0] - car.position[0])
    closing_speed = _forward_speed(car) - _forward_speed(front)
    if gap <= 0 or closing_speed <= 0:
        return MAX_TIME_TO_COLLISION
    return min(MAX_TIME_TO_COLLISION, gap / closing_speed)


def _forward_speed(car):
    return float(car.speed * math.cos(car.heading))
