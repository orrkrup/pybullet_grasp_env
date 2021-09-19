
import numpy as np
from tqdm import trange
from multiprocessing import Pool

from grasp_test import semi_random_act
from box_in_bin_env import BoxInBin


def run_trial(kd, kp):
    env = BoxInBin(use_ui=False)

    env.sim.robot.kd = kd
    env.sim.robot.kp = kp

    env.reset()
    done = False
    errors = []
    while not done:
        act = semi_random_act()
        _, _, done, info = env.step(act)
        errors.append(info['movement_error'])

    return np.mean(errors)


def mutate(kp, kd, thresh=0.5):
    if np.random.uniform() > thresh:
        kd = kd * np.random.uniform(0.5, 1.5, size=kd.size)
    if np.random.uniform() > thresh:
        kd = kd * np.random.uniform(0.5, 1.5, size=kp.size)

    return kp, kd


def sequential_trial(best_kp, best_kd):
    err_thresh = 0.05

    best_err = np.inf

    for ep in trange(100):
        if not ep:
            kp = best_kp
            kd = best_kd
        else:
            kp, kd = mutate(kp, kd)

        err = run_trial(kd, kp)

        if err < best_err:
            best_kd = kd
            best_kp = kp
            best_err = err

        if err < err_thresh:
            break

    return best_kp, best_kd, best_err


def parallel_trial(init_kp, init_kd, num_workers=8, num_iterations=100):
    kp_list = [init_kp]
    kd_list = [init_kd]
    best_kp = init_kp
    best_kd = init_kd
    best_err = np.inf

    for _ in range(num_workers - 1):
        kp, kd = mutate(kp_list[0], kd_list[0])
        kp_list.append(kp)
        kd_list.append(kd)

    for _ in trange(num_iterations):
        p = Pool(num_workers)
        errs = p.starmap(run_trial, zip(kp_list, kd_list))

        inds = np.argsort(errs)[:int(len(errs) / 2)]

        if errs[inds[0]] < best_err:
            best_err = errs[inds[0]]
            best_kp = kp_list[inds[0]]
            best_kd = kd_list[inds[0]]

        new_kps = []
        new_kds = []
        for ind, (kp, kd) in enumerate(zip(kp_list, kd_list)):
            if ind in inds:
                new_kps.append(kp)
                new_kds.append(kd)
                m_kd, m_kp = mutate(kp, kd)
                new_kps.append(m_kp)
                new_kds.append(m_kd)

        kp_list = new_kps
        kd_list = new_kds

    return best_kp, best_kd, best_err

if __name__ == '__main__':
    init_kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0, 10.0, 10.0])
    init_kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0, 10.0, 10.0])

    # sequential_trial(init_kp, init_kd)
    best_kp, best_kd, best_err = parallel_trial(init_kp, init_kd)

    print(f"KD: {best_kd}")
    print(f"KP: {best_kp}")
    print(f"error: {best_err}")