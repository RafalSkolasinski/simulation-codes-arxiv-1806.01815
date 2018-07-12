# This file contains functions connected strictly to performing numerical
# calculations on the g-factor anisotropy in the Ge/Si core/shell nanowires
# and quantum dots.


import gc
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from scipy.sparse import csc_matrix
from scipy.optimize import minimize
from types import SimpleNamespace
from functools import lru_cache

import kwant

from . import misc
from . import model
from . import shapes



# Simulation parameters
def parameters(shape, grid, R, L=None, kz=None, B=0, theta=0, phi=0,
               Ex=0, Ey=0, delta_epsilon=0, magnetism='both',
               soi='both'):
    """Define parameters for simulation.

    Parameters
    ----------
    shape : str
        Name of the shape to be used. Available shapes are defined in
        "shapes.py" file.
    grid : int of float
        Grid spacing using to discretize the system.
    R : int or float
        Radius of the wire.
    L : int, float, or 'None' (default)
        Length of the wire. If not 'None' value is provided system will be
        a finite wire (quantum dot). Mutually exclusive with "kz".
    kz : int, float, or 'None' (default)
        Momentum value along the wire. Mutually exclusive with "L".
    B : int or float, default: 0
        Magnitude of the magnetic field.
    theta : int or float, default: 0
        The theta angle of a magnetic field vector in spherical coordinates.
    phi : int or float, default: 0
        The phi angle of a magnetic field vector in spherical coordinates.
    Ex : int or float, default: 0
        Electric field along the x-direction.
    Ey : int or float, default: 0
        Electric field along the y-direction.
    delta_epsilon : int or float, default: 0
        Strength of the strain in the wire, equal to "epsilon_zz - epsilon_rr".
    magnetism : str, default: "both"
        Type of magnetism contributions present in the system.
        Possible options are ["zeeman", "orbital", "both"].
    soi : str, default: "both"
        Type of spin-orbit present in the system.
        Possible options are ["direct", "rashba", "both"].

    Returns
    -------
    (syst_pars, sim_pars) : namespaces
        Namespaces containing validated system and simulation parameters
    """

    if (L is None and kz is None) or (L is not None and kz is not None):
        raise ValueError('Value for "kz" or "L" must be provided.')

    # Check if defined contributions are valid
    if magnetism not in ['zeeman', 'orbital', 'both']:
        raise ValueError("Allowed magnetic contributions are: ",
                         "'zeeman', 'orbital' or 'both'.")

    if soi not in ['direct', 'rashba', 'both']:
        raise ValueError("Allowed spin-orbit contributions are: ",
                         "'direct', 'rashba' or 'both'.")

    syst_pars = SimpleNamespace(shape=shape, grid=grid, R=R, L=L,
                                magnetism=magnetism, soi=soi)
    sim_pars = SimpleNamespace(kz=kz, B=B, theta=theta, phi=phi, Ex=Ex, Ey=Ey,
                               delta_epsilon=delta_epsilon)
    return syst_pars, sim_pars


# Definition of simulation system

@lru_cache()
def discretize_and_fill(smp, shape, grid, R, L=None):
    """Discretize given operator and fill appropriate system."""
    # Define coords and shape function
    coords = 'xy' if L is None else 'xyz'
    shape_2D = shapes.define_shape(shape, R)

    if L is None:
        shape = shape_2D
        start = (0, 0)
    else:
        shape = lambda s: shape_2D(s) and (np.abs(s.pos[2]) < L / 2)
        start = (0, 0, 0)

    # Use modified version of "kwant.continuum.discretize" to workaround
    # flood-fill algorithm when discretizing operators.
    tb = misc.discretize_with_hoppings(
        smp, coords, grid_spacing=grid
    )
    syst = kwant.Builder()
    syst.fill(tb, shape, start);
    return syst.finalized()


@lru_cache()
def initialize_system(shape, grid, R, L=None, magnetism='both', soi='both'):
    """Initialize Hamiltonian system accordingly to a given parameters."""

    # Decode magnetic field contribution
    zeeman = magnetism in ['zeeman', 'both']
    orbital = magnetism in ['orbital', 'both']

    # Decode spin-orbit contrubtion
    direct_soi = soi in ['direct', 'both']
    rashba_soi = soi in ['rashba', 'both']

    # Get sympy Hamiltonian
    hamiltonian = model.hamiltonian(
        direct_soi=direct_soi, rashba_soi=rashba_soi, strain=True,
        orbital=orbital, zeeman=zeeman
    )

    return discretize_and_fill(hamiltonian, shape, grid, R, L)


def diagonalize(syst_pars, sim_pars, number_of_states=20, eigenvectors=False):
    """Return eigen-energies and/or eigen-vectors of specified system."""

    # Transform magnetic field to Cartesian coordinate system and
    # prepare hamiltonian parameters.
    Bx = sim_pars.B * np.sin(sim_pars.theta) * np.cos(sim_pars.phi)
    By = sim_pars.B * np.sin(sim_pars.theta) * np.sin(sim_pars.phi)
    Bz = sim_pars.B * np.cos(sim_pars.theta)

    params = {'k_z': sim_pars.kz, 'B_x': Bx, 'B_y': By, 'B_z': Bz,
              'E_x': sim_pars.Ex, 'E_y': sim_pars.Ey,
              'delta_epsilon': sim_pars.delta_epsilon,
              'exp': np.exp, **model.material_parameters}

    # Initialize system and diagonalize the matrix
    syst = initialize_system(**vars(syst_pars))
    mat = syst.hamiltonian_submatrix(params=params, sparse=True)
    ev, evec = sla.eigsh(mat, k=number_of_states, which='SA')

    indx = np.argsort(ev)
    ev, evec = ev[indx], evec[:, indx]

    # Return desired output
    if eigenvectors:
        return ev, evec
    else:
        return ev


def calculate_expectation(operator, state, syst_pars):
    """Calculate expectation value of a given operator for eigenstates.

    Parameters
    ----------
    operator : discretizer compatible input
    state : single eigen vector or whole "evec"
    syst_pars : system parameters

    Returns
    -------
    expectation value
        if "state" is many states then a sparse matrix is returned.
    """
    state = csc_matrix(state) if len(state.shape) == 2 else state
    syst = discretize_and_fill(operator, syst_pars.shape, syst_pars.grid,
                               syst_pars.R, syst_pars.L)
    mat = syst.hamiltonian_submatrix(sparse=True)
    return (state.conjugate().transpose() @ mat @ state).real


def analyse(syst_pars, sim_pars, number_of_states=20):
    """Calculate state's g-factors and expectation values of operators."""
    ev, evec = diagonalize(
        syst_pars, sim_pars, number_of_states, eigenvectors=True
    )

    expectations = {
        name: calculate_expectation(op, evec, syst_pars).diagonal()
        for name, op in model.operators.items()
    }

    if syst_pars.L is not None:
        expectations['k_z^2'] = calculate_expectation(
            'k_z**2 * eye(4)', evec, syst_pars
        ).diagonal()

    if np.allclose(sim_pars.B, 0):
        gfactors = None
    else:
        theta, phi = sim_pars.theta, sim_pars.phi
        J = (+ np.cos(theta) * expectations['J_z']
             + np.cos(phi) * np.sin(theta) * expectations['J_x']
             + np.sin(phi) * np.sin(theta) * expectations['J_y'])

        gfactors = np.sign(np.diff(J)[::2]) * np.diff(ev)[::2]
        gfactors = gfactors / (model.constants['mu_B'] * sim_pars.B)
        gfactors = [g for g in gfactors for n in range(2)]

    gc.collect()
    return {'energies': ev, 'gfactors': gfactors, **expectations}


def find_eso(syst_pars, sim_pars):
    E0 = diagonalize(syst_pars, sim_pars, number_of_states=2)[0]

    def f(kz):
        sim_pars.kz = kz
        return diagonalize(syst_pars, sim_pars, number_of_states=2)[0]

    r = minimize(f, 0)
    return E0 - r.fun
