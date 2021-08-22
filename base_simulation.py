import os
import numpy as np
import pybullet as p
import pybullet_data as pd


class BaseSimulation(object):

    def __init__(self, use_ui: bool = True,
                 image_width=128, image_height=128,
                 physics_client_id=None):

        # initiate pybullet
        if physics_client_id is not None:
            self.pcid = physics_client_id
        else:
            self.pcid = p.connect(p.GUI if use_ui else p.DIRECT)
        print(f"Pybullet simulation connected to id {self.pcid}")
        p.loadPlugin("pdControlPlugin")

        # Set basic world parameters
        p.setPhysicsEngineParameter(maxNumCmdPer1ms=1000, physicsClientId=self.pcid)
        p.setAdditionalSearchPath(pd.getDataPath(), physicsClientId=self.pcid)
        p.setGravity(0, 0, -9.8, physicsClientId=self.pcid)
        self.plane_id = self.load_plane(checker_board=True)

        # Simulation Time
        fps = 400
        self._control_dt = 1. / fps
        p.setTimeStep(self._control_dt, physicsClientId=self.pcid)
        self.t = 0.

        # Visual parameters
        self.enable_rendering = False
        render_fps = 15
        self._render_dt = 1. / render_fps
        self._render_t = 0
        self._image_width = image_width
        self._image_height = image_height

        # Observation camera parameters
        self._projection_matrix, self._view_matrix = self.calc_camera_params(distance=0.5, yaw=0., pitch=-89., fov=90)

        # UI/debug camera parameters
        self.use_ui = use_ui
        if self.use_ui:
            self.set_debug_camera()
        self.reset()

    def disconnect(self):
        p.disconnect(self.pcid)

    def reset(self) -> None:
        self._render_t = 0
        self.t = 0.

    def step_to_state(self, state: np.ndarray, max_iter: int = None, eps: float = 0.01,
                      stop_at_collision: bool = False, closed_gripper: bool = True) -> bool:
        raise NotImplementedError

    def step(self):
        """
        perform one pybullet simulation step
        """
        self._render_t += self._control_dt
        self.t += self._control_dt
        if self.enable_rendering and self._render_t >= self._render_dt:
            self.render()
            self._render_t = 0
        p.stepSimulation(physicsClientId=self.pcid)

    def calc_camera_params(self, distance: float = 0.8, yaw: float = 5., pitch: float = -45., fov: float = 80,
                   near_val: float = 0.01, far_val: float = 4., target: list = None, position: list = None):
        if target is None:
            target = [0., 0.6, -0.13]
        if position is not None:
            view_matrix = p.computeViewMatrix(cameraEyePosition=position, cameraTargetPosition=target,
                                              cameraUpVector=[0, 0, 1], physicsClientId=self.pcid)
        else:
            view_matrix = p.computeViewMatrixFromYawPitchRoll(distance=distance, yaw=yaw, pitch=pitch, upAxisIndex=2,
                                                              cameraTargetPosition=target, roll=0,
                                                              physicsClientId=self.pcid)
        projection_matrix = p.computeProjectionMatrixFOV(fov=fov, aspect=1., nearVal=near_val, farVal=far_val,
                                                         physicsClientId=self.pcid)
        return projection_matrix, view_matrix

    def set_debug_camera(self, distance=1.2, yaw=0, pitch=-45, position=None):
        if position is None:
            position = [0., 0.6, -0.13]
        p.resetDebugVisualizerCamera(cameraDistance=distance, cameraYaw=yaw, cameraPitch=pitch,
                                     cameraTargetPosition=position, physicsClientId=self.pcid)

    def render(self, return_seg_mask=False, return_depth_img=False, view_matrix=None, projection_matrix=None):
        """
        render pybullet scene
        :return: a rendered scene image
        """
        if view_matrix is None or projection_matrix is None:
            view_matrix = self._view_matrix
            projection_matrix = self._projection_matrix

        img_data = p.getCameraImage(self._image_width,
                                    self._image_height,
                                    view_matrix,
                                    projection_matrix,
                                    shadow=True,
                                    flags=p.ER_SEGMENTATION_MASK_OBJECT_AND_LINKINDEX,
                                    physicsClientId=self.pcid)

        width, height, rgb_img, depth_img, seg_img = img_data

        rendered_img = np.reshape(rgb_img, (height, width, 4))[:, :, :3]
        ret_dict = {'rgb': rendered_img}

        if return_seg_mask:
            seg_img = np.reshape(seg_img, (height, width, 1))
            ret_dict['seg'] = seg_img

        if return_depth_img:
            depth_img = np.reshape(depth_img, (height, width, 1))
            ret_dict['dep'] = depth_img

        return ret_dict

    def change_object_mass(self, object_id: int, mass: float, change_interia: bool = True) -> None:
        if change_interia:
            dimensions = p.getVisualShapeData(object_id, physicsClientId=self.pcid)[0][3]
            inertia = [(1 / 12) * mass * (dimensions[1] ** 2 + dimensions[2] ** 2),
                       (1 / 12) * mass * (dimensions[0] ** 2 + dimensions[2] ** 2),
                       (1 / 12) * mass * (dimensions[0] ** 2 + dimensions[1] ** 2)]
            p.changeDynamics(object_id, -1, mass=mass, localInertiaDiagonal=inertia, physicsClientId=self.pcid)
        else:
            p.changeDynamics(object_id, -1, mass=mass, physicsClientId=self.pcid)

    def change_object_friction(self, object_id: int, friction: float) -> None:
        p.changeDynamics(object_id, -1, lateralFriction=friction, physicsClientId=self.pcid)

    def let_objects_settle(self, steps=50) -> None:
        # let object fall and settle
        for i in range(steps):
            self.step()

    def is_object_moving(self, obj_id):
        return np.linalg.norm(self.get_object_velocity(obj_id)) > self.velocity_sensitivity

    def get_object_velocity(self, obj_id) -> np.ndarray:
        return np.array(p.getBaseVelocity(obj_id, physicsClientId=self.pcid), dtype=float)

    def get_object_state(self, obj_id) -> np.ndarray:
        state = p.getBasePositionAndOrientation(obj_id, physicsClientId=self.pcid)
        return np.concatenate((state[0], state[1]))

    def get_object_orientation(self, obj_id) -> np.ndarray:
        return self.get_object_state(obj_id)[3:7]

    def get_object_position(self, obj_id) -> np.ndarray:
        return self.get_object_state(obj_id)[:3]

    def get_object_xz_position(self, obj_id) -> np.ndarray:
        return self.get_object_state(obj_id)[[0, 2]]

    def load_plane(self, checker_board: bool = False) -> int:
        flags = p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        base_orn = p.getQuaternionFromEuler([0, 0, 0])
        plane_path = os.path.join('plane.urdf')
        return p.loadURDF(plane_path, basePosition=[0, 0, 0], baseOrientation=base_orn, flags=flags,
                          physicsClientId=self.pcid)
