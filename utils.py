import numpy as np


def get_quaternion_from_matrix(M):
    tm = np.trace(M)
    q0 = np.sqrt((tm + 1.) / 4)
    st = (1. - tm) / 4.
    m = lambda ind, p: (ind + p) % 3
    q = []
    for ind in range(3):
        arg = M[ind, ind] / 2. + st
        if np.isclose(arg, 0.):
            arg = 0.
        q.append(np.sign(M[m(ind, 2), m(ind, 1)] - M[m(ind, 1), m(ind, 2)]) * np.sqrt(arg))

    q.append(q0)
    return np.array(q)


def quatdiff_in_euler(quat_curr, quat_des):
    # Adapted from https://github.com/justagist/pybullet_robot/blob/46ab382a36649c33ed1d61a34c5dfbcc753c8e08/src/pybullet_robot/controllers/utils.py#L4
    curr_mat = np.array(p.getMatrixFromQuaternion(quat_curr)).reshape((3, 3))
    des_mat = np.array(p.getMatrixFromQuaternion(quat_des)).reshape((3, 3))

    rel_mat = des_mat.T.dot(curr_mat)

    rel_quat = get_quaternion_from_matrix(rel_mat)

    return np.array(p.getEulerFromQuaternion(rel_quat))


def interpolate_quaternions(curr_orn, goal_orn, t):
    theta = 2 * np.arccos(curr_orn.dot(goal_orn))
    return (np.sin((1.0 - t) * theta) * curr_orn + np.sin(t * theta) * goal_orn) / np.sin(theta)


class PolyTraj(object):
    def __init__(self, q0, qf, tf, qd0=0., qdf=0., qdd0=0., qddf=0., t0=0.):
        pol = lambda t: np.array([[t ** k for k in range(6)],
                                  [0] + [ind * t ** (k - 1) for ind, k in enumerate(range(1, 6))],
                                  [0, 0, 2, 6 * t, 12 * t ** 2, 20 * t ** 3]])
        mat = np.concatenate((pol(t0), pol(tf)), axis=0)
        vec = np.array([q0, qd0, qdd0, qf, qdf, qddf])

        self.coefs = np.linalg.lstsq(mat, vec, rcond=None)[0]
        self.tf = tf

    def get_q_qd(self, t):
        p = np.polynomial.Polynomial(self.coefs)
        return p(t), p.deriv(1)(t)
