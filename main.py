import unsio.output
import numpy as np


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
