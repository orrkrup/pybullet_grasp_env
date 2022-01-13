import logging

import numpy as np
from gym.spaces import Box
from gym import core
from grasp_simulation import GraspSimulation


class BoxInBin(core.Env):
    def __init__(self, trial_len=10, use_ui=True):
        self.sim = GraspSimulation(imp_control=False, use_ui=use_ui)
        self.action_space = Box(low=-np.pi/2., high=np.pi/2., shape=(6, 2))

        self.observation_space = Box(low=0, high=1.0, shape=(self.sim._image_height, self.sim._image_width, 3),
                                     dtype=np.float)

        self.err_thresh = 0.01
        self.trial_len = trial_len
        self.trial_reward = 0
        self.step_counter = 0
        self.obs = None
        self.bin_id = self.sim.load_bin(friction=1.2)
        self.cubeset_id = None
        self.num_cubes = 7

    def reset(self):
        self.trial_reward = 0
        self.step_counter = 0

        self.sim.robot.reset()
        self.sim.clear_objects()
        self.cubeset_id = self.sim.load_cubeset(self.num_cubes, mode='grid_cut')
        self.sim.let_objects_settle(steps=100)

        images = self.sim.render(return_depth_img=True)
        self.obs = np.concatenate((images['rgb'], images['dep']), axis=-1)

        return self.obs

    def step(self, act):
        # pre action
        er1 = self.sim.step_to_state(act[0], closed_gripper=False)

        # action
        er2 = self.sim.step_to_state(act[1], closed_gripper=False)
        self.sim.grasp()

        # Pick object up to reset position, gripper closed
        er3 = self.sim.move_to_reset(closed_gripper=True)
        r = 1.0 if self.sim.check_object_height() else 0.0
        self.sim.release()

        self.sim.let_objects_settle(steps=100)

        self.step_counter += 1

        images = self.sim.render(return_depth_img=True)
        self.obs = np.concatenate((images['rgb'], images['dep']), axis=-1)

        info_dict = {"movement_error": np.mean((er1, er2, er3))}

        return self.obs, r, self.step_counter >= self.trial_len, info_dict

    def render(self, mode='human'):
        return self.obs

    def close(self):
        self.sim.disconnect()