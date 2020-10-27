import numpy as np
import scipy.constants
import unsio.output
from collections import namedtuple
from tqdm import tqdm

G = scipy.constants.G
TIME_INTERVAL = 5

Body = namedtuple('Body', 'name, mass, position, velocity')


def main():
    def rd_vector(): return np.random.normal(0.5, 0.2, (3,))

    bodies = (
        Body(name='P1', mass=1, position=rd_vector(), velocity=rd_vector()),
        Body(name='P2', mass=1 / 4, position=rd_vector(), velocity=rd_vector())
    )

    # arr = create_random(len(bodies))
    arr = np.array(tuple((b.position, b.velocity) for b in bodies))
    # output = unsio.output.CUNS_OUT("computation.nemo", 'nemo', float32=False)
    for _ in tqdm(range(100)):
        for _ in range(1000):
            calc_next_step(arr, bodies)
        # output.setData(arr, 'stars', 'pos')
    print(arr)


"""
Utility functions to create the position and velocity arrays for bodies either randomly ('normal') or from predefined 
bodies

Format of the returned arrays: 
Numpy Float64 array with 3 axes (dimensions):
* List of Bodies
* List of [Position, Velocity]
* List of Coordinates (3DVector, 3*float64)
"""
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
        vel = np.zeros((3,))
        for idx2, b2 in enumerate(arr):

            if idx1 == idx2:
                continue

            m1, m2 = bodies[idx1].mass, bodies[idx2].mass

            dist_vector = (b1[0] - b2[0]) ** 2
            normalized_v = dist_vector / np.linalg.norm(dist_vector)
            r_2 = np.sum(dist_vector)

            F = G * m2 / r_2
            vel += normalized_v * F
        arr[idx1, 1] = vel
    for idx, b in enumerate(arr):
        arr[idx, 0] = b[0] + b[1] * TIME_INTERVAL


if __name__ == '__main__':
    main()
