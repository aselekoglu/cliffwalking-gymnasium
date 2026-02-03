import gymnasium
import cliffwalking_env
from stable_baselines3 import DQN, PPO
import time

env = gymnasium.make("cliffwalking_env/CliffWalking-v0")

print("Starting DQN training...")
model = DQN("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=1)
print("Done training!")
model.save("dqn_cliffwalking")

print("Starting PPO training...")
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=10000, log_interval=1)
print("Done training!")
model.save("ppo_cliffwalking")

del model # remove to demonstrate saving and loading

env = gymnasium.make("cliffwalking_env/CliffWalking-v0", render_mode="human")


model = DQN.load("dqn_cliffwalking")
# model = PPO.load("ppo_cliffwalking")
print(f"Loaded model: {model}")
obs, info = env.reset()

while True:
    action, _states = model.predict(obs, deterministic=False)

    # SB3 returns a NumPy array; env expects a scalar action.
    action = int(action) if hasattr(action, "item") else action
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        # Small pause so terminal steps are visible in the renderer.
        time.sleep(0.2)
        obs, info = env.reset()
