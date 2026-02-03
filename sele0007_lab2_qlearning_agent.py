import gymnasium
import cliffwalking_env
import numpy as np
import time
import os
import matplotlib.pyplot as plt

# create environment
env = gymnasium.make("cliffwalking_env/CliffWalking-v0", render_mode="human")
observation, info = env.reset()

# QTable : contains the Q-Values for every (state,action) pair
grid_width = int(env.observation_space["agent"].high[0]) + 1
grid_height = int(env.observation_space["agent"].high[1]) + 1
numstates = grid_width * grid_height
numactions = env.action_space.n
qtable = np.random.rand(numstates, numactions).tolist()

# hyperparameters
episodes = 35
gamma = 0.1
epsilon = 0.08
decay = 0.1
alpha = 1

# training loop
episode_returns = {}
episode_steps = {}
for i in range(episodes):
    state_dict, info = env.reset()
    state = state_dict['agent'][1] * (env.observation_space['agent'].high[0] + 1) + state_dict['agent'][0]
    steps = 0
    # track total reward
    total_reward = 0

    done = False
    while not done:
        if os.name == "nt" or os.environ.get("TERM"):
            os.system("cls" if os.name == "nt" else "clear")
        # display total reward
        env.render()
        time.sleep(0.01)
        # count steps to finish game
        steps += 1

        # act randomly sometimes to allow exploration
        if np.random.uniform() < epsilon:
            action = env.action_space.sample()
        # if not select max action in Qtable (act greedy)
        else:
            action = qtable[state].index(max(qtable[state]))

        # take action
        next_state_dict, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        # add reward to total reward
        total_reward += reward
        next_state = next_state_dict['agent'][1] * grid_width + next_state_dict['agent'][0]
        # update qtable value with Bellman equation
        qtable[state][action] = reward + gamma * max(qtable[next_state])

        # update state
        state = next_state


    # The more we learn, the less we take random actions
    epsilon -= decay*epsilon

    # Add a new entry to the dictionary with the current episode's return and steps
    episode_returns[i] = total_reward
    episode_steps[i] = steps
    print(f"Episode {i+1}:")
    print(f"  Reward: {total_reward}")
    print(f"  Steps: {steps}")
    print("---------------------------------------")
    time.sleep(0.3)

# add matplotlib code to plot the episode returns (cumulative total reward) and the steps per episode
plt.plot(list(episode_returns.values()), label="Rewards per Episode", marker='o', linewidth=2, )
plt.plot(list(episode_steps.values()), label="Steps per Episode", marker='o', linewidth=2,)
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.minorticks_on()
plt.legend()
plt.title("Training Results")
plt.grid(which="major", alpha=0.75)
plt.grid(which="minor", alpha=0.2)
plt.show()
