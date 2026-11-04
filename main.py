from src.cooperative_driving.environments.customEnv import CustomHighwayEnv


def main():
    env = CustomHighwayEnv(render_mode="human")

    observation, info = env.reset(seed=0)

    for step in range(100):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if step % 10 == 0:
            kinematics = observation[0]
            print("Step:", step)
            print("First vehicle:", kinematics[0])

        if terminated or truncated:
            observation, info = env.reset()

    env.close()


if __name__ == "__main__":
    main()