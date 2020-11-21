import pyopencl as cl
import os
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


def write_to_unsio(output, arr, bodies, t):
    output.setData(np.arange(0, len(bodies), dtype=np.int32), 'stars', 'id')
    output.setData(np.array([b.mass for b in bodies]), 'stars', 'mass')
    output.setData(t, 'time')
    output.setData(arr[:, [0]].flatten(), 'stars', 'pos')
    output.setData(arr[:, [1]].flatten(), 'stars', 'vel')


def pretty_format_bytes(size):
    for x in ['bytes', 'KB', 'MB', 'GB', 'TB']:
        if size < 1024.0:
            return "%3f %s" % (size, x)
        size /= 1024.0
    return str(size)


def pretty_print_prop_dict(values):
    return "   - ".join(": ".join((key, val)) for key, val in values.items())


def show_info(platforms):
    print(f"Found {len(platforms)} OpenCL Platform(s):")
    for p in platforms:
        print(f' * Platform: {p.name} (', pretty_print_prop_dict(
            {"Vendor": p.vendor, "Version": p.version, "Profile": p.profile}
        ), ')')
        for device in p.get_devices():
            print(f'\t* Device: {device.name} (', pretty_print_prop_dict(
                {
                    "Vendor": device.vendor,
                    "Driver Version": device.driver_version,
                    "Profile": device.profile,
                    "Version": device.version,
                    "Global Memory Size": pretty_format_bytes(device.global_mem_size),
                    "Local Memory Size": pretty_format_bytes(device.local_mem_size),
                    "Max Work Group Size": str(device.max_work_group_size)
                }
            ), ')')
    print()


if __name__ == '__main__':
    platforms = cl.get_platforms()
    show_info(platforms)

    # This allows sharing memory between multiple processes (without having to use blocking queues or other
    # IPC mechanisms. This should be the fastest way and it is all abstracted as a Numpy array.
    bodies = tuple(get_random_bodies(40000))
    arr = create_arr_from_bodies(bodies)
    masses = np.array([b.mass for b in bodies])

    print(arr)

    ctx = cl.create_some_context(interactive=True)
    # ctx = cl.Context(platforms[0].get_devices())
    queue = cl.CommandQueue(ctx)
    mf = cl.mem_flags

    a_g = cl.Buffer(ctx, mf.READ_WRITE | mf.USE_HOST_PTR, hostbuf=arr)
    m_g = cl.Buffer(ctx, mf.READ_ONLY | mf.USE_HOST_PTR, hostbuf=masses)

    with open("related/newton_test.cl", "r") as f:
        code = f.read()

    for x in range(100):
        prg = cl.Program(ctx, code).build()
        prg.new_vel(queue, (len(bodies),), None, a_g, m_g, np.int32(len(bodies)))
        queue.finish()
    print(arr)
