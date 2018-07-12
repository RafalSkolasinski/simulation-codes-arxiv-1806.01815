# Definition of Luttinger-Kohn Hamiltonian.
# Model defined in this file describes dispersion
# of heavy and light hole states in Si/Ge shell/core nanowires.
# The shell effect is included only through the strain that it
# induces in the core. Implementation assumes circular symmetry
# of the Hamiltonian.


import scipy.constants
from scipy.constants import physical_constants
from functools import lru_cache

import kwant
import sympy


###### Constants and material parameters
epsilon_rr = + 3.5e-2
epsilon_zz = - 1.5e-2
delta_epsilon = epsilon_zz - epsilon_rr

constants = {
    'm_0': scipy.constants.m_e / scipy.constants.e / (1e9)**2,
    'phi_0': 2 * physical_constants['mag. flux quantum'][0] * (1e9)**2,
    'mu_B': physical_constants['Bohr magneton in eV/T'][0],
    'hbar': scipy.constants.hbar / scipy.constants.eV,
}

material_parameters = {
    'gamma_1': 13.35,
    'gamma_2': 4.25,   # Not used
    'gamma_3': 5.69,   # Not used
    'gamma_s': 5.114,  # Defined as: (2 * gamma_2 + 3 * gamma_3 ) / 5

    'kappa': 3.41,
    'alpha': -0.4,

    'b': -2.5,
    'd': -5.0,         # Not used

    **constants
}


###### Magnetic field
orbital_effect = {
    kwant.continuum.sympify(k): kwant.continuum.sympify(v)
    for k, v in [
        ("k_x", "k_x + (2 * pi / phi_0) * (- B_z * y / 2)"),
        ("k_y", "k_y + (2 * pi / phi_0) * (+ B_z * x / 2)"),
        ("k_z", "k_z + (2 * pi / phi_0) * (B_x * y - B_y * x)"),
    ]
}

vector_potential = "[-B_z * y / 2, B_z * x / 2, B_x * y - B_y * x]"


###### Circular approximation
circular_approximation = {
    sympy.sympify('gamma_2'): sympy.sympify('gamma_s'),
    sympy.sympify('gamma_3'): sympy.sympify('gamma_s'),
    sympy.sympify('d'): sympy.sympify('sqrt(3) * b'),
    sympy.sympify('epsilon_xx'): sympy.sympify('epsilon_rr'),
    sympy.sympify('epsilon_yy'): sympy.sympify('epsilon_rr'),
}


###### Spin-3/2 angular momentum matrices and non-commutative symbols
Jx = sympy.Rational(1, 2) * sympy.Matrix([[0, sympy.sqrt(3), 0, 0],
                                          [sympy.sqrt(3), 0, 2, 0],
                                          [0, 2, 0, sympy.sqrt(3)],
                                          [0, 0, sympy.sqrt(3), 0]])

Jy = sympy.I * sympy.Rational(1, 2) * sympy.Matrix([[0, -sympy.sqrt(3), 0, 0],
                                                    [sympy.sqrt(3), 0, -2, 0],
                                                    [0, 2, 0, -sympy.sqrt(3)],
                                                    [0, 0, sympy.sqrt(3), 0]])

Jz = sympy.Rational(1, 2) * sympy.diag(3, 1, -1, -3)

matrix_locals = {'I_4x4': sympy.eye(4), 'J_x': Jx, 'J_y': Jy, 'J_z': Jz}

j_locals = {name: sympy.Symbol(name, commutative=False)
            for name in ['J_x', 'J_y', 'J_z']}


###### Model components (direction dependent)
component_luttinger_kohn = kwant.continuum.sympify("""
    hbar**2 / (2 * m_0) * (
        + (gamma_1 + (5/2) * gamma_2) * (k_x**2 + k_y**2 + k_z**2) * I_4x4
        - 2 * gamma_2 * (k_x**2 * J_x**2 + k_y**2 * J_y**2 + k_z**2 * J_z**2)
        - gamma_3 * (+ (k_x * k_y + k_y * k_x) * (J_x * J_y + J_y * J_x)
                     + (k_y * k_z + k_z * k_y) * (J_y * J_z + J_z * J_y)
                     + (k_z * k_x + k_x * k_z) * (J_z * J_x + J_x * J_z)))""",
    locals=j_locals
)

component_rashba_soi = kwant.continuum.sympify("""
    alpha * (+ E_x * (k_y * J_z - k_z * J_y)
             + E_y * (k_z * J_x - k_x * J_z)
             + E_z * (k_x * J_y - k_y * J_x))""",
    locals=j_locals
)


###### Model components (direction independent)
component_direct_soi = kwant.continuum.sympify(
    """-(E_x * x + E_y * y + E_z * z) * I_4x4"""
)

# delta_epsilon = epsilon_zz - epsilon_rr
component_strain = kwant.continuum.sympify(
    """b * delta_epsilon * J_z**2""",
)

component_zeeman = kwant.continuum.sympify(
    """2 * kappa * mu_B * (B_x * J_x + B_y * J_y + B_z * J_z)""",
)


###### Define the total Hamiltonian
@lru_cache()
def hamiltonian(direct_soi=True, rashba_soi=True, strain=True, orbital=True,
                zeeman=True):
    """Define the Luttinger-Kohn Hamiltonian with specified components."""

    # Direction dependent components
    smp = component_luttinger_kohn

    if rashba_soi:
        smp += component_rashba_soi

    # Direction independent components
    if strain:
        smp += component_strain

    if direct_soi:
        smp += component_direct_soi

    # Add magnetic field contributions
    if zeeman:
        smp += component_zeeman

    if orbital:
        smp = smp.subs(orbital_effect)

    # Apply circular approximation and do final cleaning
    smp = smp.subs(circular_approximation)
    smp = smp.subs({kwant.continuum.sympify(name): 0 for name in ['z', 'E_z']})
    smp = kwant.continuum.sympify(str(smp), locals=matrix_locals)

    return smp.expand()


###### Definition of operators that can be used to calculate expectation values
operators = {
    name: kwant.continuum.sympify(operator, locals=matrix_locals)
    for name, operator in [
        ("J_z", "J_z"),
        ("J_x", "J_x"),
        ("J_y", "J_y"),
        ("L_z", "(x * k_y - y * k_x) * eye(4)"),
        ("F_z", "(x * k_y - y * k_x) * eye(4) + J_z"),
        ("LH", "diag(0, 1, 1, 0)"),
    ]
}
