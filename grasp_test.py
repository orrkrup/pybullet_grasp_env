
import numpy as np
import math
from box_in_bin_env import BoxInBin
import pybullet as p


def cube_act(loc, orn=None):
    act = np.zeros(7)
    act[0] = loc[0]
    act[1] = loc[1]
    act[2] = loc[2]
    if orn is None:
        act[3:7] = p.getQuaternionFromEuler([math.pi / 2., 0.0, 0.0])
        # act[3:7] = p.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.0])
    else:
        e = p.getEulerFromQuaternion(orn)
        act[3:7] = act[3:7] = p.getQuaternionFromEuler([math.pi / 2., e[1], 0.0])
    return act


def bin_act():
    act = np.zeros(7)
    act[0] = 0.29
    act[1] = 0.14
    act[2] = -0.6
    act[3:7] = p.getQuaternionFromEuler([math.pi / 2., math.pi / 2., 0.0])
    return act


def random_act():
    act = np.zeros((2, 7))
    act[:, 0] = np.random.uniform(-0.27, 0.27, size=(2,))
    act[:, 1] = np.random.uniform(0.01, 0.1, size=(2,))
    act[:, 2] = np.random.uniform(-0.39, -0.81, size=(2,))
    for ind in range(2):
        z1 = np.random.uniform(np.pi / 4., 3 * np.pi / 4.)
        y1 = np.random.uniform(-np.pi, np.pi)
        act[ind, 3:7] = p.getQuaternionFromEuler((z1, y1, 0.0))
    print(act[0])
    print(act[1])
    return act


def semi_random_act():
    act = np.zeros((2, 7))
    act[:, 0] = np.random.uniform(-0.27, 0.27, size=(2,))
    act[:, 1] = np.random.uniform(0.39, 0.81, size=(2,))
    act[:, 2] = np.random.uniform(0.01, 0.14, size=(2,))
    act[0, 2] = 0.2
    z1 = np.random.uniform(-np.pi / 2., np.pi / 2.)
    for ind in range(2):
        act[ind, 3:7] = p.getQuaternionFromEuler((np.pi, 0.0, z1))
    return act


def test_act():
    act = np.zeros((2, 7))
    act[:, 0] = np.array((-0.2, 0.24))
    act[:, 1] = np.array((0.6, 0.6))
    act[:, 2] = np.array((0.04, 0.04))
    for ind in range(2):
        z1 = np.pi
        y1 = 0.0
        act[ind, 3:7] = p.getQuaternionFromEuler((z1, 0, 0))
    return act


if __name__ == '__main__':
    # env = PandaGrasp()
    env = BoxInBin()

    for ep in range(7):
        rewards = 0
        obs = env.reset()
        done = False
        while not done:
            if np.random.uniform() > 0.0:
                act = semi_random_act()
                # act = test_act()
            else:
                act = random_act()

            obs, r, done, info = env.step(act)
            print(obs.shape)
            if r > 0:
                print("Success")
            rewards += r

        print('Done, reward: {}'.format(rewards))

