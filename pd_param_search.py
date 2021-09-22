
import numpy as np
from tqdm import trange
from multiprocessing import Pool

from grasp_test import semi_random_act
from box_in_bin_env import BoxInBin


def run_trial(kp, kd):
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

    env.close()
    return np.mean(errors)


def mutate(kp, kd, thresh=0.5, uniform=False):
    if uniform:
        per_kp = kp * np.random.uniform(0.5, 1.5, size=kp.size)
        per_kd = kd * np.random.uniform(0.5, 1.5, size=kd.size)
    else:
        per_kp = np.clip(np.random.normal(loc=kp, scale=np.sqrt(kp)), 0.0, None)
        per_kd = np.clip(np.random.normal(loc=kd, scale=np.sqrt(kd)), 0.0, None)

    dec_val = np.random.uniform()
    if dec_val < thresh / 2.:
        return kp, per_kd
    elif dec_val < thresh:
        return per_kp, kd
    else:
        return per_kp, per_kd


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

    try:
        with trange(num_iterations) as t:
            for _ in t:
                pl = Pool(num_workers)
                errs = pl.starmap(run_trial, zip(kp_list, kd_list))
                pl.close()

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
                        m_kp, m_kd = mutate(kp, kd)
                        new_kps.append(m_kp)
                        new_kds.append(m_kd)

                kp_list = new_kps
                kd_list = new_kds

                t.set_postfix(err=best_err)
    finally:
        return best_kp, best_kd, best_err


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--workers', type=int, default=8, help='Number of workers to spawn')
    parser.add_argument('--generations', type=int, default=1000, help='Number of evolutionary iterations to run')
    args = parser.parse_args()

    init_kd = np.array([50.0, 50.0, 50.0, 50.0, 30.0, 25.0, 15.0, 10.0, 10.0])
    init_kp = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0, 10.0, 10.0])

    # sequential_trial(init_kp, init_kd)
    best_kp, best_kd, best_err = parallel_trial(init_kp, init_kd, num_workers=args.workers,
                                                num_iterations=args.generations)
    print(f"KD: {best_kd}")
    print(f"KP: {best_kp}")
    print(f"error: {best_err}")