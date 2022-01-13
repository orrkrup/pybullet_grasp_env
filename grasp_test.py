
import logging
logging.basicConfig(level=logging.DEBUG)

from box_in_bin_env import BoxInBin
from actions import *


if __name__ == '__main__':
    env = BoxInBin()

    for ep in range(7):
        rewards = 0
        obs = env.reset()
        done = False
        while not done:
            if np.random.uniform() > 1.0:
                act = semi_random_act()
            else:
                act = rand_cube_act(env)

            obs, r, done, info = env.step(act)
            if r > 0:
                print("Success")
            rewards += r

        print('Done, reward: {}'.format(rewards))

