import gym
import sconegym

# create the sconegym env
env = gym.make("sconegait2d-v0")

for episode in range(100):
    # store the results of every 10th episode
    # storing results is slow, and should only be done sparsely
    # stored results can be analyzed in SCONE Studio
    if episode % 10 == 0:
        env.store_next_episode()

    episode_steps = 0
    total_reward = 0
    state = env.reset()

    while True:
        # samples random action
        action = env.action_space.sample()

        # applies action and advances environment by one step
        next_state, reward, done, info = env.step(action)

        episode_steps += 1
        total_reward += reward

        # to render results, open a .sto file in SCONE Studio
        #env.render()

        # check if done
        if done or (episode_steps >= 1000):
            print(f'Episode {episode} finished; steps={episode_steps}; reward={total_reward:0.3f}')
            break
        
env.close()
