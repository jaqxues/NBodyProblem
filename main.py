from random import random
import numpy as np
import scipy.constants
import unsio.output
from collections import namedtuple

G = scipy.constants.G
TIME_INTERVAL = 5

Body = namedtuple('Body', 'name, mass, position, velocity')


def main():
    print("Run as main only for debugging purposes. Please use the ipynb to run the simulation")
    bodies = tuple(get_random_bodies(4))
    arr = create_arr_from_bodies(bodies)
    pretty_print_coordinates(bodies, arr)

    print(arr)
    calc_next_step(arr, bodies)
    print(arr)
    for _ in range(10):
        calc_next_step(arr, bodies)
    print(arr)


def get_random_bodies(n=2):
    def rd_vector(mu): return np.random.normal(mu, 0.2, (3,))

    for i in range(n):
        yield Body(name=f'P{i}', mass=100 * random(), position=rd_vector(0.5), velocity=np.zeros((3,)))


"""
Utility functions to create the position and velocity arrays for bodies either randomly ('normal') or from predefined 
bodies

Format of the returned arrays: 
Numpy Float64 array with 3 axes (dimensions):
* List of Bodies
* List of [Position, Velocity]
* List of Coordinates (3DVector, 3*float64)
"""


def pretty_print_coordinates(bodies, arr):
    for body, x in zip(bodies, arr):
        p = x[0]
        coordinates = '; '.join(f'{p[i]:.4f}' for i in range(3))
        print(f'{body.name} - Current Position ({coordinates})')


def create_arr_from_bodies(bodies): return np.array(tuple((b.position, b.velocity) for b in bodies))
def create_random(amount): return np.random.normal(0.5, 0.3, (amount, 2, 3))


def calc_next_step(arr, bodies):
    """
    1. Calculate norm of the vector a:
        * Newton's law of motion:
            ``F = m * a
        * Newton's law of universal gravitation:
            ``F = G* ((m1 * m2) / (r^2))``
        --> ``m1 * a = G * ((m1 * m2) / (r^2))``
            ``a = G * (m2 / r^2)``
    2. Direction of the vector a:
        Normalized vector `r`.
    :param arr: Array with velocities and positions
    :return: Return array with updated velocities and positions
    """
    for idx1, b1 in enumerate(arr):
        acceleration = np.zeros((3,))
        for idx2, b2 in enumerate(arr):

            if idx1 == idx2:
                continue

            d_vec = np.subtract(b2[0], b1[0])
            normalized = d_vec / np.linalg.norm(d_vec)

            r_2 = np.sum(np.power(d_vec, 2))
            m2 = bodies[idx2].mass

            a = G * m2 / r_2
            acceleration += (a * normalized)
        arr[idx1, 1] += acceleration * TIME_INTERVAL
    for idx, b in enumerate(arr):
        arr[idx, 0] += b[1] * TIME_INTERVAL


def write_unsio_initial(output, bodies):
    output.setData(np.arange(0, len(bodies), dtype=np.int32), 'stars', 'id')
    output.setData(np.array([b.mass for b in bodies]), 'stars', 'mass')


def unsio_example():
    output = unsio.output.CUNS_OUT("computation.g2", 'gadget2', float32=False)
    # Write
    output.save()
    output.close()


def write_to_unsio(output, arr, t):
    output.setData(t, 'time')
    output.setData(arr[:, [0]].flatten(), 'stars', 'pos')
    output.setData(arr[:, [1]].flatten(), 'stars', 'vel')


if __name__ == '__main__':
    main()
