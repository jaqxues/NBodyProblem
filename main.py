import multiprocessing as mp
import os
from tqdm import tqdm
from random import random
from scipy.constants import G
import unsio.output
import numpy as np
from collections import namedtuple

TIMESTEP = 2


def newton_parallelized(ifrom, ito):
    for i1 in range(ifrom, ito):
        b1_pos = arr[i1][0]
        acceleration = np.zeros((3,))
        for i2, b2_pos in enumerate(map(lambda x: x[0], arr)):

            if i1 == i2:
                continue

            d_vec = np.subtract(b2_pos, b1_pos)
            normalized = d_vec / np.linalg.norm(d_vec)

            r_2 = np.sum(np.power(d_vec, 2))
            m2 = bodies[i2].mass

            a = G * m2 / r_2
            acceleration += (a * normalized)
        arr[i1, 1] += acceleration * TIMESTEP


def compute_multi_process(pool, steps, indices):
    for _ in range(steps):
        pool.starmap(newton_parallelized, indices)
        # if chunked > 500:
        #     pass  # TODO Start in mp Pool
        # else:
        a.acquire()
        for b in range(len(arr)):
            arr[b, 0] += arr[b, 1] * TIMESTEP
        a.release()


Body = namedtuple('Body', 'name, mass, position, velocity')


def create_arr_from_bodies(bodies, arr=None):
    if arr is None:
        return np.array(tuple((b.position, b.velocity) for b in bodies))
    for ib, b in enumerate(bodies):
        arr[ib] = b.position, b.velocity
    return arr


def get_random_bodies(n=2):
    '''
    Generates `n` bodies randomly, with a random mass, position and 0 velocity.
    '''

    def rd_vector(mu): return np.random.normal(mu, 0.2, (3,))

    for i in range(n):
        yield Body(name=f'P{i}', mass=100 * random(), position=rd_vector(0.5), velocity=np.zeros((3,)))


def gen_indices():
    total = len(bodies)

    chunked = total // threads
    bonus = total - chunked * threads

    for x in range(threads):
        sub_total = chunked
        if x + 1 == threads:
            sub_total += bonus
        ifrom = chunked * x
        yield ifrom, ifrom + sub_total


def write_to_unsio(output, arr, bodies, t):
    output.setData(np.arange(0, len(bodies), dtype=np.int32), 'stars', 'id')
    output.setData(np.array([b.mass for b in bodies]), 'stars', 'mass')
    output.setData(t, 'time')
    output.setData(arr[:, [0]].flatten(), 'stars', 'pos')
    output.setData(arr[:, [1]].flatten(), 'stars', 'vel')


if __name__ == '__main__':
    # This allows sharing memory between multiple processes (without having to use blocking queues or other
    # IPC mechanisms. This should be the fastest way and it is all abstracted as a Numpy array.
    bodies = tuple(get_random_bodies(400))
    a = mp.Array(np.ctypeslib.as_ctypes_type(np.float64), len(bodies) * 6)
    arr = create_arr_from_bodies(bodies, np.frombuffer(a.get_obj()).reshape((len(bodies), 2, 3)))

    os.makedirs("output", exist_ok=True)

    threads = mp.cpu_count()
    indices = tuple(gen_indices())

    with mp.Pool(processes=threads) as pool:
        for idx in tqdm(range(200)):
            try:
                compute_multi_process(pool, 20, indices)
                output = unsio.output.CUNS_OUT(f"output/computation{idx}", 'gadget2', float32=False)
                write_to_unsio(output, arr, bodies, idx)
                output.save()
                output.close()
            except KeyboardInterrupt:
                print("Interrupting Computation")
                break

    print("Generating Ascii File")
    with open("output/file", "w+") as f:
        for x in range(idx):
            f.write(f"computation{x}\n")
