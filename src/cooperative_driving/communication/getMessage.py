def getMessage(car):
    """
    Return the current information shared by a vehicle.
    """
    return {
        "vehicle_id": car.vehicle_id,
        "position": car.position.copy(),
        "velocity": car.velocity.copy(),
        "action": car.action.copy(),
        "intent": car.intent.copy(),
    }
