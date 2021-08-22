import numpy as np
import pybullet as p
from gym.spaces import Box
from gym import core
from grasp_simulation import GraspSimulation


class BoxInBin(core.Env):
    def __init__(self, trial_len=10):
        self.sim = GraspSimulation(imp_control=True)
        self.action_space = Box(low=-np.pi/2., high=np.pi/2., shape=(6,))

        self.observation_space = Box(low=0, high=1.0, shape=(self.sim._image_height, self.sim._image_width, 3),
                                     dtype=np.float)

        self.trial_len = trial_len
        self.trial_reward = 0
        self.step_counter = 0
        self.obs = None
        self.bin_id = self.sim.load_bin(friction=1.2)
        self.cubeset_id = None

    def reset(self):
        self.trial_reward = 0
        self.step_counter = 0

        self.sim.robot.reset()
        self.sim.clear_objects()
        self.cubeset_id = self.sim.load_cubeset(7, mode='grid_cut')
        self.sim.let_objects_settle(steps=100)

        self.obs = self.sim.render()

        return self.obs

    def step(self, act):
        # pre action
        self.sim.step_to_state(act[0], closed_gripper=False)

        #action
        self.sim.step_to_state(act[1], closed_gripper=False)
        self.sim.grasp()

        # # TODO: probably not what we want the robot to do with the object
        # vertical_offset = np.zeros_like(act[1])
        # vertical_offset[2] += 0.2
        # post_act = act[1] + 2 * vertical_offset
        # post_act[3:7] = p.getQuaternionFromEuler((np.pi, 0., 0.))
        # self.sim.step_to_state(post_act, closed_gripper=True)
        #
        self.sim.move_to_reset(closed_gripper=True)
        r = 1.0 if self.sim.check_object_height() else 0.0
        self.sim.release()

        self.sim.let_objects_settle(steps=100)

        self.step_counter += 1

        images = self.sim.render(return_depth_img=True)
        self.obs = np.concatenate((images['rgb'], images['dep']), axis=-1)

        info_dict = {}

        return self.obs, r, self.step_counter >= self.trial_len, info_dict

    def render(self, mode='human'):
        return self.obs