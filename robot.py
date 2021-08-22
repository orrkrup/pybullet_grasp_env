import numpy as np
import pybullet as p


class PandaRobot(object):
    def __init__(self, physics_client_id: int):

        # init bullet client
        self.pcid = physics_client_id
        p.setPhysicsEngineParameter(solverResidualThreshold=0, physicsClientId=self.pcid)
        self.gravity_vector = np.array([0.0, 0.0, -9.8])

        self.position_sensitivity = 0.005
        self.velocity_sensitivity = 0.0001

        # init panda robot
        orn = p.getQuaternionFromEuler([0, 0, np.pi / 2])
        self._robot_id = p.loadURDF("franka_panda/panda.urdf", np.array([0, 0, 0]), orn, useFixedBase=True,
                                    flags=p.URDF_USE_SELF_COLLISION | p.URDF_USE_SELF_COLLISION_EXCLUDE_PARENT,
                                    physicsClientId=self.pcid)

        self.control_mode = 'pos'
        self._num_dofs = 7

        p.setCollisionFilterPair(self._robot_id, self._robot_id, 6, 8, 0, physicsClientId=self.pcid)
        p.setCollisionFilterPair(self._robot_id, self._robot_id, 9, 10, 0, physicsClientId=self.pcid)

        self._end_effector_index = 11
        self.reset_joint_positions = [-0.017792060227770554, -0.7601235411041661, 0.019782607023391807,
                                      -2.342050140544315, 0.029840531355804868, 1.5411935298621688, 0.7534486589746342,
                                      0.04, 0.04]

        # Joint Information
        self.n_joints = p.getNumJoints(self._robot_id, physicsClientId=self.pcid)
        joint_prop_names = ['jointName', 'jointType', 'jointLowerLimit', 'jointUpperLimit', 'jointMaxForce',
                            'jointMaxVelocity']

        joint_props = self._get_joint_properties(properties=joint_prop_names)

        self.movable_joints = [ind for ind, t in enumerate(joint_props['jointType']) if p.JOINT_FIXED != t]
        self.n_movable_joints = len(self.movable_joints)

        self.joint_lower_bounds = np.array([joint_props['jointLowerLimit'][k] for k in self.movable_joints]) * 10.
        self.joint_upper_bounds = np.array([joint_props['jointUpperLimit'][k] for k in self.movable_joints]) * 10.
        self.joint_torque_limits = np.array([joint_props['jointMaxForce'][k] for k in self.movable_joints])
        self.joint_vel_limits = np.array([joint_props['jointMaxVelocity'][k] for k in self.movable_joints])
        self.joint_ranges = [10 * k for k in [5.8, 3.5, 5.8, 3.1, 5.8, 3.8, 5.8, 0.04, 0.04]]
        self.joint_frictions = [5.0, 2.0, 2.0, 0.5, 1.0, 0.5, 0.5, 0.0, 0.0]
        self.joint_damping = [10.0, 5.0, 5.0, 1.0, 2.0, 1.0, 1.0, 0.0, 0.0]

        self.link_com_pos = [p.getLinkState(self._robot_id, link_index, computeLinkVelocity=1,
                                            computeForwardKinematics=1, physicsClientId=self.pcid)[2] for link_index
                             in range(self.n_joints)]

        for joint, dmp, fric in zip(self.movable_joints, self.joint_damping, self.joint_frictions):
            p.changeDynamics(self._robot_id, joint, contactDamping=dmp, contactStiffness=fric, physicsClientId=self.pcid)

        self.next_torques = None
        self.last_gravity_comp = None

        # Gripper parameters
        self.finger_inds = self.movable_joints[-2:]
        self.open_finger_positions = self.joint_upper_bounds[-2:]
        self.closed_finger_positions = self.joint_lower_bounds[-2:]

        # Optionally update gripper properties
        # p.changeDynamics(self.robot_id, 9, lateralFriction=5.0)
        # p.changeDynamics(self.robot_id, 10, lateralFriction=5.0)

        # p.enableJointForceTorqueSensor(self.robot_id, 9, True)
        # p.enableJointForceTorqueSensor(self.robot_id, 10, True)

        # Constrain finger joints to mirror each other
        c = p.createConstraint(self._robot_id,
                               9,
                               self._robot_id,
                               10,
                               jointType=p.JOINT_GEAR,
                               jointAxis=[1, 0, 0],
                               parentFramePosition=[0, 0, 0],
                               childFramePosition=[0, 0, 0])
        p.changeConstraint(c, gearRatio=-1, erp=0.1, maxForce=50, physicsClientId=self.pcid)

        self.reset()

    def step(self):
        if 'tor' in self.control_mode:
            if self.last_gravity_comp is None:
                self.compute_gravity_comp()
            tau_g = self.last_gravity_comp

            if self.next_torques is not None:
                cmd = self.next_torques + tau_g
                self.last_gravity_comp = None
            else:
                cmd = tau_g

            p.setJointMotorControlArray(self._robot_id, self.movable_joints, controlMode=p.TORQUE_CONTROL,
                                        forces=np.clip(cmd.squeeze(), -self.joint_torque_limits, self.joint_torque_limits),
                                        physicsClientId=self.pcid)

            self.next_torques = None

    def _get_joint_properties(self, properties=None, joints=None):
        prop_ind_to_name = ['jointIndex', 'jointName', 'jointType', 'qIndex', 'uIndex', 'flags', 'jointDamping',
                            'jointFriction', 'jointLowerLimit', 'jointUpperLimit', 'jointMaxForce','jointMaxVelocity',
                            'linkName', 'jointAxis', 'parentFramePos', 'parentFrameOrn', 'parentIndex']

        if joints is None:
            joints = range(self.n_joints)

        if properties is None:
            properties = prop_ind_to_name

        joint_info = list(zip(*[p.getJointInfo(self._robot_id, i, physicsClientId=self.pcid) for i in joints]))

        return {prop_name: joint_info[prop_ind_to_name.index(prop_name)] for prop_name in properties}

    def is_close(self, target_joints, source_joints=None):
        distance = self.get_distance(target_joints, source_joints)
        return distance < self.position_sensitivity

    def get_distance(self, target_joints, source_joints=None):
        assert len(target_joints) == self.n_movable_joints
        if source_joints is None:
            source_joints = self.get_robot_state()[0]
        assert len(source_joints) == self.n_movable_joints
        return np.linalg.norm(np.array(source_joints) - np.array(target_joints))

    def is_moving(self):
        current_speed = self.get_current_speed()
        return current_speed > self.velocity_sensitivity

    def set_torque_control(self):
        p.setJointMotorControlArray(self._robot_id, self.movable_joints, p.VELOCITY_CONTROL,
                                    forces=[0.0] * self.n_movable_joints,
                                    physicsClientId=self.pcid)
        self.control_mode = 'tor'

    def set_joint_torques(self, cmd):
        assert len(cmd) == len(self.movable_joints)
        self.next_torques = cmd

    def get_jacobian(self, joint_angles=None, link_index=None):
        """
        :return: Jacobian matrix for provided joint configuration
        :rtype: ndarray (shape: 6x9)
        :param joint_angles: Optional parameter. If different than None,
                             then returned jacobian will be evaluated at
                             given joint_angles. Otherwise the returned
                             jacobian will be evaluated at current robot
                             joint angles.
        :param link_index: Optional parameter denoting the link for which to calculate the Jacobian.
                           Calculated at CoM by default.
        :type joint_angles: [float] * len(self.get_movable_joints)
        """

        if joint_angles is None:
            joint_angles = list(zip(*p.getJointStates(self._robot_id, self.movable_joints,
                                                      physicsClientId=self.pcid)))[0]

        if link_index is None:
            link_index = self._end_effector_index

        linear_jacobian, angular_jacobian = p.calculateJacobian(bodyUniqueId=self._robot_id,
                                                                linkIndex=link_index,
                                                                localPosition=self.link_com_pos[link_index],
                                                                objPositions=list(joint_angles),
                                                                objVelocities=[0] * self.n_movable_joints,
                                                                objAccelerations=[0] * self.n_movable_joints,
                                                                physicsClientId=self.pcid)

        rot_mat = np.array(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(self._robot_id)[1])).reshape((3, 3))
        jacobian = np.vstack([rot_mat.dot(np.array(linear_jacobian)), rot_mat.dot(np.array(angular_jacobian))])

        return jacobian

    def compute_gravity_comp(self):

        mass_link_indices = list(range(self.n_joints))
        jp_list = [self.get_jacobian(link_index=ind)[:3] for ind in mass_link_indices]
        m_list = [p.getDynamicsInfo(self._robot_id, ind, physicsClientId=self.pcid)[0] for ind in mass_link_indices]

        g = np.sum([m * jp.T.dot(self.gravity_vector) for m, jp in zip(m_list, jp_list)], axis=0)

        self.last_gravity_comp = -g

    def get_robot_state(self):
        joint_position_velocity_pairs = [
            (t[0], t[1])
            for t in p.getJointStates(self._robot_id, self.movable_joints, physicsClientId=self.pcid)
        ]
        return list(zip(*joint_position_velocity_pairs))

    def get_current_speed(self):
        velocities = self.get_robot_state()[1]
        return np.linalg.norm(velocities)

    def set_xyz(self, state: np.ndarray, closed_gripper=True, max_force=None) -> None:
        position = state[:3]
        orientation = state[3:7] if len(state) > 3 else None
        joint_positions = self.inverse_kinematics(position=position, orientation=orientation)
        if closed_gripper:
            joint_positions = tuple(list(joint_positions[:-2]) + [0., 0.])
        else:
            gripper_pos = np.array(self.get_robot_state()[0][-2:])
            joint_positions = tuple(list(joint_positions[:-2]) + list(gripper_pos))

        self.set_joints(joint_positions, max_force)

    def set_joints(self, joint_positions: np.ndarray, max_force=None) -> None:
        control_mode = p.POSITION_CONTROL
        if max_force is None:
            max_force = self.joint_torque_limits
        # else:
        #     max_force = np.minimum(np.abs(max_force), self.joint_torque_limits)
        p.setJointMotorControlArray(self._robot_id,
                                    jointIndices=self.movable_joints,
                                    controlMode=control_mode,
                                    targetPositions=joint_positions,
                                    targetVelocities=[0.01] * (self.n_movable_joints - 2) + [0.0, 0.0],
                                    forces=max_force,
                                    positionGains=[0.01] * (self.n_movable_joints - 2) + [1., 1.],
                                    velocityGains=[0.8] * (self.n_movable_joints - 2) + [1., 1.],
                                    physicsClientId=self.pcid)

    def inverse_kinematics(self, position: np.ndarray, orientation: np.ndarray = None, max_iter: int = 50) -> np.ndarray:
        if orientation is None:
            joint_positions = p.calculateInverseKinematics(self._robot_id,
                                                           endEffectorLinkIndex=self._end_effector_index,
                                                           targetPosition=position,
                                                           lowerLimits=self.joint_lower_bounds.tolist(),
                                                           upperLimits=self.joint_upper_bounds.tolist(),
                                                           jointRanges=self.joint_ranges,
                                                           restPoses=self.reset_joint_positions,
                                                           maxNumIterations=max_iter,
                                                           physicsClientId=self.pcid)
        else:
            joint_positions = p.calculateInverseKinematics(self._robot_id,
                                                           endEffectorLinkIndex=self._end_effector_index,
                                                           targetPosition=position,
                                                           targetOrientation=orientation,
                                                           lowerLimits=self.joint_lower_bounds.tolist(),
                                                           upperLimits=self.joint_upper_bounds.tolist(),
                                                           jointRanges=self.joint_ranges,
                                                           restPoses=self.reset_joint_positions,
                                                           maxNumIterations=max_iter,
                                                           physicsClientId=self.pcid)
        return joint_positions

    def reset(self, joint_positions=None) -> None:
        if joint_positions is None:
            joint_positions = self.reset_joint_positions

        for index, j in enumerate(self.movable_joints):
            p.resetJointState(self._robot_id, j, joint_positions[index], physicsClientId=self.pcid, targetVelocity=0.0)
            p.setJointMotorControl2(self._robot_id, j, p.POSITION_CONTROL,
                                    targetPosition=joint_positions[index],
                                    positionGain=0.2, velocityGain=1.0,
                                    physicsClientId=self.pcid)

    def get_ee_state(self):
        state = p.getLinkState(self._robot_id, self._end_effector_index,
                               computeLinkVelocity=1,
                               computeForwardKinematics=1,
                               physicsClientId=self.pcid)

        # Cartesian 6D pose:
        pos = state[4]
        orn = state[5]

        return np.array(pos), np.array(orn)

    def get_ee_vel(self):
        ee_state = p.getLinkState(self._robot_id, self._end_effector_index, computeLinkVelocity=1,
                                  computeForwardKinematics=1, physicsClientId=self.pcid)
        velocity = ee_state[-2]
        angular_velocity = ee_state[-1]
        return np.array(velocity), np.array(angular_velocity)

    def get_link_poses(self):
        link_poses = []
        for i in self.movable_joints:
            state = p.getLinkState(self._robot_id, i, physicsClientId=self.pcid)
            position = state[4]
            orientation = state[5]
            link_poses.append((position, orientation))
        return link_poses

    def get_collisions(self):
        return [contact for contact in p.getContactPoints(self._robot_id, physicsClientId=self.pcid) if
                contact[8] < -0.0001]

    def is_collision(self):
        collisions = self.get_collisions()
        return len(collisions) > 0

    def open_gripper(self) -> None:
        p.setJointMotorControlArray(self._robot_id,
                                    jointIndices=self.finger_inds,
                                    controlMode=p.POSITION_CONTROL,
                                    targetPositions=self.open_finger_positions,
                                    physicsClientId=self.pcid)

    def close_gripper(self, action=None) -> None:
        # move finger joints in position control
        if action is None:
            action = [0.01, 0.01]

        # Can't use p.setJointMotorControlArray since maxVelocity is critical for successful grasping
        for finger, act in zip(self.finger_inds, action):
            p.setJointMotorControl2(self._robot_id, finger, p.POSITION_CONTROL, targetPosition=act, force=10.,
                                    maxVelocity=0.1, physicsClientId=self.pcid)

    def check_contact_fingertips(self, obj_id):
        # check if there is any contact on the internal part of the fingers, to control if they are correctly touching an object

        # idx_fingers = [self._joint_name_to_ids['panda_finger_joint1'], self._joint_name_to_ids['panda_finger_joint2']]
        idx_fingers = [9, 10]

        p0 = p.getContactPoints(obj_id, self.pcid, linkIndexB=idx_fingers[0],
                                physicsClientId=self.pcid)
        p1 = p.getContactPoints(obj_id, self.pcid, linkIndexB=idx_fingers[1],
                                physicsClientId=self.pcid)

        p0_contact = 0
        p0_f = [0]
        if len(p0) > 0:
            # get cartesian position of the finger link frame in world coordinates
            w_pos_f0 = p.getLinkState(self.pcid, idx_fingers[0], physicsClientId=self.pcid)[4:6]
            f0_pos_w = p.invertTransform(w_pos_f0[0], w_pos_f0[1])

            for pp in p0:
                # compute relative position of the contact point wrt the finger link frame
                f0_pos_pp = p.multiplyTransforms(f0_pos_w[0], f0_pos_w[1], pp[6], f0_pos_w[1])

                # check if contact in the internal part of finger
                if f0_pos_pp[0][1] <= 0.001 and f0_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p0_contact += 1
                    p0_f.append(pp[9])

        p0_f_mean = np.mean(p0_f)

        p1_contact = 0
        p1_f = [0]
        if len(p1) > 0:
            w_pos_f1 = p.getLinkState(self.pcid, idx_fingers[1], physicsClientId=self.pcid)[4:6]
            f1_pos_w = p.invertTransform(w_pos_f1[0], w_pos_f1[1])

            for pp in p1:
                # compute relative position of the contact point wrt the finger link frame
                f1_pos_pp = p.multiplyTransforms(f1_pos_w[0], f1_pos_w[1], pp[6], f1_pos_w[1])

                # check if contact in the internal part of finger
                if f1_pos_pp[0][1] >= -0.001 and f1_pos_pp[0][2] < 0.055 and pp[8] > -0.005:
                    p1_contact += 1
                    p1_f.append(pp[9])

        p1_f_mean = np.mean(p0_f)

        return (p0_contact > 0) + (p1_contact > 0), (p0_f_mean, p1_f_mean)