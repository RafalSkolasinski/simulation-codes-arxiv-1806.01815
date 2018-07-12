# Function defined in this file serve miscellaneous purposes.
# See comments for each group of functions for more details.


import numpy as np
import xarray as xr
from itertools import product
from collections import Mapping, defaultdict

import sympy
import kwant


# Code below is to workaround "flood-fill" algorithm that does not
# fill systems with missing hoppings.

def discretize_with_hoppings(hamiltonian, coords=None, *, grid_spacing=1,
                             locals=None):
    """Discretize system and add zero-magnitude hoppings where required.

    This is modification of the "kwant.continuum.discretize" function
    that adds zero-magnitude hoppings in place of missing ones.

    Please check "kwant.continuum.discretize" documentation for details.
    """
    template = kwant.continuum.discretize(hamiltonian, coords,
                                          grid_spacing=grid_spacing,
                                          locals=locals)

    syst = kwant.Builder(template.symmetry)
    lat = template.lattice

    syst[next(iter(template.sites()))] = np.zeros((lat.norbs, lat.norbs))
    syst[lat.neighbors()] = np.zeros((lat.norbs, lat.norbs))

    syst.update(template)
    return syst


# Function defined in this section come from "kwant.continuum" module
# of Kwant and are currently a part of a non-public API.
# To avoid breakage with future releases, they are defined here.

def make_commutative(expr, *symbols):
    """Make sure that specified symbols are defined as commutative.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
    symbols: sequace of symbols
        Set of symbols that are requiered to be commutative. It doesn't matter
        of symbol is provided as commutative or not.

    Returns
    -------
    input expression with all specified symbols changed to commutative.
    """
    names = [s.name if not isinstance(s, str) else s for s in symbols]
    symbols = [sympy.Symbol(name, commutative=False) for name in names]
    expr = expr.subs({s: sympy.Symbol(s.name) for s in symbols})
    return expr


def monomials(expr, gens=None):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
        Sympy expression to be parsed into monomials.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of monomials. If unset it will default to all
        symbols used in ``expr``.

    Returns
    -------
    dictionary (generator: monomial)

    Example
    -------
        >>> expr = kwant.continuum.sympify("A * (x**2 + y) + B * x + C")
        >>> monomials(expr, gens=('x', 'y'))
        {1: C, x: B, x**2: A, y: A}
    """
    if gens is None:
        gens = expr.atoms(sympy.Symbol)
    else:
        gens = [kwant.continuum.sympify(g) for g in gens]

    if not isinstance(expr, sympy.MatrixBase):
        return _expression_monomials(expr, gens)
    else:
        output = defaultdict(lambda: sympy.zeros(*expr.shape))
        for (i, j), e in np.ndenumerate(expr):
            mons = _expression_monomials(e, gens)
            for key, val in mons.items():
                output[key][i, j] += val
        return dict(output)


def _expression_monomials(expr, gens):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr
        Sympy expr to be parsed.
    gens: sequence of sympy.Symbol
        Generators of monomials.

    Returns
    -------
    dictionary (generator: monomial)
    """
    expr = sympy.expand(expr)
    output = defaultdict(lambda: sympy.Integer(0))
    for summand in expr.as_ordered_terms():
        key = []
        val = []
        for factor in summand.as_ordered_factors():
            symbol, exponent = factor.as_base_exp()
            if symbol in gens:
                key.append(factor)
            else:
                val.append(factor)
        output[sympy.Mul(*key)] += sympy.Mul(*val)

    return dict(output)



# Various helpers for handling simulation and data.
# This collection of functions helps to organize, combine and plot outputs
# of the simulation.

def reduce_dimensions(data):
    """Reduce dimensions of length equal to 1."""
    sel = {k: data.coords[k].data[0]
           for k, v in data.dims.items() if v==1}

    data = data.sel(**sel)
    return data


def dict_product(**parameters):
    """Compute Cartesian product of named sets."""
    output = [{k: v for k, v in zip(parameters.keys(), x)}
              for x in list(product(*parameters.values()))]
    return output


def serialize_none(x):
    """Substitute None with its string representation."""
    return str(x) if x is None else x


def to_xarray(coords, data_vars):
    """Represent single simulation as xarray DataSet."""
    coords = {k: serialize_none(v) for k, v in coords.items()}
    ds = xr.Dataset(data_vars, coords)

    #Assign coordinates to dimensions that misses them
    assignment = {dim: ds[dim] for dim in set(ds.dims) - set(ds.coords)}
    ds = ds.assign_coords(**assignment)

    return ds


def combine_datasets(sets, dims=None):
    """Combine datasets along specified dimension.

    If "dims" is None it will default to all dimensions
    that are not used as coordinates.
    """
    ds = xr.concat(sets, dim='internal', coords='all')

    if dims is None:
        dims = list(set(ds.coords) - set(ds.dims))

    ds = ds.set_index(internal=list(dims))
    ds = ds.unstack('internal')
    return ds


def iterate(dataset, dims):
    """Iterate over all xarray dimensions except specified ones.

    Parameters
    ----------
    dataset : xarray Dataset
        Input dataset
    dims : sequence of strings
        Dimension to exclude from iteration

    Returns
    -------
    names : sequence of strings
        Names of coordinates that are being iterated.

    iterator : iterator over (key, par, val)
        key : sequence of values in the same order as returned by "names"
        val : subset of dataset corresponding to iteration
    """
    names = [p for p in list(dataset.dims) if p not in dims]
    stacked = dataset.stack(internal_iterator=names)
    stacked = stacked.transpose('internal_iterator', *dims)

    def iterator():
        for i, p in enumerate(stacked.internal_iterator):
            key = p.data.tolist()
            val = stacked.sel(internal_iterator=p).drop('internal_iterator')
            val = val.assign_coords(**dict(zip(names, key)))
            yield (key, val)

    return names, iterator()
