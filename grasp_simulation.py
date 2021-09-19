import logging
import numpy as np
import pybullet as p
from base_simulation import BaseSimulation
from robot import PandaRobot
from utils import PolyTraj

logging.basicConfig(level=logging.DEBUG)


class GraspSimulation(BaseSimulation):
    def __init__(self, imp_control=True, use_ui=True):
        super(GraspSimulation, self).__init__(use_ui=use_ui)

        # Robot
        self.robot = PandaRobot(physics_client_id=self.pcid)
        self.robot.reset()
        self.reset_ee_state = np.concatenate(self.robot.get_ee_state())

        # Controller Parameters
        self._max_move_steps = 500
        self.error_thresh = 0.005

        if imp_control:
            self.imp_control = True
            self.start_err = np.asarray([200., 200.])
            self.robot.set_torque_control()
        else:
            self.imp_control = False

        # Object Details
        self.object_id = None
        self.object_height_thresh = self.reset_ee_state[2] - 0.1
        self.object_base_pos = np.array([0.0, 0.6, 0.3])
        self.bin_pos = np.array([0.0, 0.6, 0.085])

        # Visual details
        if self.use_ui:
            self.set_debug_camera(yaw=180)

    def step(self):
        self.robot.step()
        super(GraspSimulation, self).step()

    def release(self):
        if self.imp_control:
            for ind in range(100):
                tau = np.zeros(self.robot.n_movable_joints)
                tau[-2:] += 5.0
                self.robot.set_joint_torques(tau)
                self.step()
        else:
            self.robot.open_gripper()
            self.let_objects_settle(100)

    def move_to_reset(self, closed_gripper=False):
        target_q = self.robot.reset_joint_positions
        if closed_gripper:
            target_q[-2:] = [0.0, 0.0]
        self.step_to_joint_state(target_q)

    def check_object_height(self, obj_id=None, height=None):
        if obj_id is None:
            obj_id = self.object_id
        if height is None:
            height = self.object_height_thresh
        object_state, _ = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.pcid)
        return object_state[2] > height

    def load_bin(self, scale: float = 1., friction: float = 2., fixed: bool = True) -> int:
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        bin_id = p.loadURDF("./objects/bin.urdf", self.bin_pos, p.getQuaternionFromEuler((0, 0, np.pi / 2)),
                            flags=flags, globalScaling=scale, physicsClientId=self.pcid)
        self.change_object_friction(bin_id, friction=friction)
        if fixed:
            self.change_object_mass(bin_id, mass=0.0, change_interia=False)
        return bin_id

    def load_cubeset(self, num_cubes, max_size=0.035, color=None, mass=None, friction=None, mode='grid_cut'):

        # Create cube shapes and sizes
        if 'random_linear' in mode:
            cube_sizes = np.tile(np.random.uniform(0.01, max_size, size=(num_cubes,)), (3, 1)).transpose()
            # cube_sizes = np.linspace(0.01, 0.035, 5)

            link_pos = [[size + cube_sizes[:, 0][ind + 1], 0, 0] for ind, size in enumerate(cube_sizes[:, 0][:-1])]

        elif 'grid_cut' in mode:
            grid = np.zeros((num_cubes, num_cubes, num_cubes))
            cube_indices = [[0, 0, 0]]
            grid[tuple(cube_indices[0])] = 1
            dir_vecs = []
            for ind in range(1, num_cubes):
                new_inds = cube_indices[-1]
                for direction in np.random.permutation(6):
                    dir_vec = np.zeros(3, dtype=np.int)
                    dir_vec[direction % 3] = (direction % 2) * 2 - 1
                    tmp_inds = np.mod(new_inds + dir_vec, num_cubes)
                    if not grid[tuple(tmp_inds)]:
                        new_inds = tmp_inds
                        grid[tuple(new_inds)] = 1
                        cube_indices.append(new_inds)
                        dir_vecs.append(dir_vec)
                        break
            assert len(cube_indices) == num_cubes

            grid_sizes = np.random.uniform(0.01, max_size, size=(num_cubes, 3))
            cube_sizes = np.choose(cube_indices, grid_sizes)
            link_pos = [dir_vec * (shape + cube_sizes[ind + 1]) for ind, (shape, dir_vec) in
                        enumerate(zip(cube_sizes[:-1], dir_vecs))]

        # Cube colors
        if color is None:
            cube_colors = [[int(b) for b in format(ind % 8, '03b')] + [1] for ind in range(num_cubes)]
        else:
            cube_colors = [color] * num_cubes

        # Create visual and collision meshes
        link_col_ids = []
        link_vis_ids = []
        for ind in range(num_cubes):
            link_col_ids.append(p.createCollisionShape(p.GEOM_BOX, halfExtents=cube_sizes[ind],
                                                       physicsClientId=self.pcid))
            link_vis_ids.append(p.createVisualShape(p.GEOM_BOX, halfExtents=cube_sizes[ind], rgbaColor=cube_colors[ind],
                                                    physicsClientId=self.pcid))

        # Update masses
        if mass is None:
            masses = np.random.uniform(0.1, 1, size=num_cubes)
        elif isinstance(mass, (tuple, list, np.ndarray)):
            assert len(mass) == num_cubes
            masses = mass
        else:
            masses = [mass] * num_cubes

        # Create multibody
        base_pos = self.object_base_pos
        self.object_id = p.createMultiBody(baseMass=masses[0], baseCollisionShapeIndex=link_col_ids[0],
                                           baseVisualShapeIndex=link_vis_ids[0], basePosition=base_pos,
                                           baseOrientation=[0, 0, 0, 1],
                                           linkMasses=masses[1:],
                                           linkCollisionShapeIndices=link_col_ids[1:],
                                           linkVisualShapeIndices=link_vis_ids[1:],
                                           linkInertialFramePositions=([[0, 0, 0]] * (num_cubes - 1)),
                                           linkInertialFrameOrientations=([[0, 0, 0, 1]] * (num_cubes - 1)),
                                           linkPositions=link_pos,
                                           linkOrientations=([[0, 0, 0, 1]] * (num_cubes - 1)),
                                           linkParentIndices=[ind for ind in range(num_cubes - 1)],
                                           linkJointTypes=[p.JOINT_FIXED] * (num_cubes - 1),
                                           linkJointAxis=([[1, 0, 0]] * (num_cubes - 1)),
                                           physicsClientId=self.pcid)

        # Update friction parameters
        if friction is None:
            link_frictions = np.random.uniform(0.4, 1.5, size=num_cubes)
        elif isinstance(friction, (tuple, list, np.ndarray)):
            assert len(friction) == num_cubes
            link_frictions = friction
        else:
            link_frictions = [friction] * num_cubes
        for ind, fric in enumerate(link_frictions):
            p.changeDynamics(self.object_id, ind, lateralFriction=fric, physicsClientId=self.pcid)

        return self.object_id

    def clear_objects(self):
        if self.object_id is not None:
            p.removeBody(self.object_id, physicsClientId=self.pcid)
            self.object_id = None

    def render(self, save_render=True, return_seg_mask=False, return_depth_img=False, view_matrix=None,
               projection_matrix=None, alternative=False):
        ret = super(GraspSimulation, self).render(return_seg_mask=return_seg_mask, return_depth_img=return_depth_img)
        ret['rgb'] = ret['rgb'] / 255.
        return ret

    def pos_grasp(self, action=None, obj_id=None):
        self.robot.close_gripper(action=action)

        for ind in range(100):
            self.step()

            # use object id if available to check contact force and eventually stop the finger motion
            if obj_id is not None:
                _, forces = self.robot.check_contact_fingertips(obj_id)
                # print("contact forces {}".format(forces))

                if forces[0] >= 20.0:
                    action[0] = self.robot.get_robot_state()[self.robot.finger_inds[0]][0]

                if forces[1] >= 20.0:
                    action[1] = self.robot.get_robot_state()[self.robot.finger_inds[0]][0]

    def grasp(self):
        if not self.imp_control:
            return self.pos_grasp()
        else:
            logging.debug('grasping...')
            for ind in range(100):
                tau = np.zeros(self.robot.n_movable_joints)
                tau[-2:] -= 5.0
                self.robot.set_joint_torques(tau)
                self.step()
            logging.debug('done grasping')
            return 0

    def pos_step_to_state(self, state: np.ndarray, max_iter: int = None, eps: float = 0.01,
                          stop_at_collision: bool = False, closed_gripper: bool = True) -> bool:
        """
        perform motion planning and execute, reaching from current state to defined state
        :param state: a goal state to reach
        :param max_iter: max number of iteration to perform
        :param eps: Euclidean distance tolerance for reaching to goal state
        :param stop_at_collision: whether to stop when colliding with some object
        :param closed_gripper: whether to ensure close gripper while moving
        :return bool: whether a collision has happened
        """
        target_pos = np.array(state[:3], dtype=float)
        max_iter = self._max_move_steps if max_iter is None else max_iter
        curr_pos = self.robot.get_ee_state()[0]
        step_counter = 1
        is_collision = False

        while np.linalg.norm(curr_pos - target_pos, 2) > eps:
            self.robot.set_xyz(state, closed_gripper=closed_gripper)
            self.step()
            curr_pos = self.robot.get_ee_state()[0]
            is_collision = self.robot.is_collision()
            if stop_at_collision and is_collision:
                break
            if step_counter >= max_iter:
                break
            step_counter += 1

        logging.debug(f"iterations: {step_counter}, target: {state[:3]}, reached: {curr_pos}, "
                      f"error: {100 * np.linalg.norm(curr_pos - state[:3])}")
        return is_collision

    def step_to_state(self, state: np.ndarray, max_iter: int = None, eps: float = 0.01,
                      stop_at_collision: bool = False, closed_gripper: bool = True,
                      visualize_goal: bool = True) -> bool:
        """
        perform motion planning and execute, reaching from current state to defined state
        :param state: a goal state to reach
        :param max_iter: max number of iteration to perform
        :param eps: Euclidean distance tolerance for reaching to goal state
        :param stop_at_collision: whether to stop when colliding with some object
        :param closed_gripper: whether to ensure close gripper while moving
        :param visualize_goal: whether to show a little red circle at current gripper target position
        :return bool: whether a collision has happened
        """
        goal_pos = np.array(state[:3], dtype=float)
        goal_orn = state[3:7]

        targ_q = np.array(self.robot.inverse_kinematics(goal_pos, goal_orn))
        targ_q_dot = np.zeros_like(targ_q)

        if visualize_goal:
            vis = p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 0.5],
                                      physicsClientId=self.pcid)
            sid = p.createMultiBody(baseVisualShapeIndex=vis, basePosition=goal_pos, physicsClientId=self.pcid)
        else:
            sid = None

        self.step_to_joint_state(targ_q, goal_vel=targ_q_dot, closed_gripper=closed_gripper)

        if sid is not None:
            p.removeBody(sid, physicsClientId=self.pcid)

    def step_to_joint_state(self, goal_state, goal_vel=None, max_iter=None, closed_gripper=False):
        """
        perform motion planning and execute, reaching from current joint state to defined joint state
        :param goal_state: a goal state to reach
        :param goal_vel: optional: goal velocity for joints
        :param max_iter: maximum number of iterations to perform
        :return bool: whether a collision has happened
        """
        curr_q, _ = self.robot.get_robot_state()
        targ_q = np.array(goal_state)
        if closed_gripper:
            targ_q[-2:] = self.robot.closed_finger_positions
        else:
            targ_q[-2:] = self.robot.open_finger_positions
        targ_q_dot = np.zeros_like(targ_q) if goal_vel is None else goal_vel

        max_iter = self._max_move_steps if max_iter is None else max_iter

        self.robot.set_torque_control()

        error = np.linalg.norm(curr_q - targ_q)
        steps = 0

        while error > self.error_thresh:

            steps += 1
            if steps > max_iter:
                break

            curr_q, q_dot = self.robot.get_robot_state()
            error = np.linalg.norm(curr_q - targ_q)

            if self.imp_control:
                delta_q = targ_q - curr_q
                delta_q_dot = q_dot - targ_q_dot

                tau = self.robot.kp * delta_q - self.robot.kd * delta_q_dot

                self.robot.set_joint_torques(tau)
            else:
                self.robot.set_joints(targ_q, joint_velocities=targ_q_dot, control_mode=p.PD_CONTROL)

            self.step()

        logging.debug(f"{'Reached Goal' if error < self.error_thresh else 'Failed'}, error: {error}")
