# Collection of shape functions used in this projects


import numpy as np
from functools import lru_cache


@lru_cache()
def define_shape(name, R):
    """Define shape of a given type and specified radius.

    Parameters
    ----------
    shape : string
        Name of the chosen shape. Options are 'hexagon', 'circle', 'square',
        or 'rectangular'.
    R : int or float
        Radius for the shape, i.e. in case of "square" it will mean half of
        the side's length. For "rectangular" it must be a tuple.

    Returns
    -------
    shape function to be used with ``.fill`` method of Kwant's builders.
    """
    if name not in ['hexagon', 'circle', 'square', 'rectangular']:
        raise ValueError('Wrong type of shape: "{}".'.format(name))

    if name == 'rectangular' and not isinstance(R, tuple):
        raise ValueError('If shape is "rectangular" then "R" must be a tuple.')

    if name == 'hexagon':
        shape = define_hexagon(R)

    if name == 'circle':
        shape = define_circle(R)

    if name == 'square':
        shape = define_square(R)

    if name == 'rectangular':
        shape = define_rectangular(R[0], R[1])

    return shape


def define_hexagon(R):
    """Return shape function for hexagon."""
    def shape(site):
        x1, x2 = np.array(site.pos)[:2]

        a0 = 0.5*R
        b0 = np.sin(np.pi/3.0)*R

        return (x2 >- b0 and x2 < b0 and
                x2 > -(b0/a0) * (2*a0 - x1) and
                x2 < -(b0/a0) * (x1 - 2*a0) and
                x2 < (b0/a0) * (x1 + 2*a0) and
                x2 > -(b0/a0) * (x1 + 2*a0))

    return shape


def define_circle(R):
    """Return shape function for circle."""
    def shape(site):
        return site.pos[0]**2 + site.pos[1]**2 < R**2

    return shape


def define_square(W):
    """Return shape function for square."""
    def shape(site):
        return np.abs(site.pos[0]) < W and np.abs(site.pos[1]) < W

    return shape


def define_rectangular(a, b):
    """Return shape function for rectangular."""
    def shape(site):
        return np.abs(site.pos[0]) < a and np.abs(site.pos[1]) < b

    return shape
