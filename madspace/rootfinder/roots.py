import torch
from torch import Tensor
from .methods import newton
from typing import Mapping, Callable, Union, Dict, List
from .autograd import RootFinderPolynomial, RootFinderMass


def get_u_parameter(xs: Tensor) -> Tensor:
    """Returns the solution of the equation
    ...math::
        r_{i-1} =(n+1-i)*u_i^{2n-2i} - (n-i)*u_i^{2n+2-2i} for u_i

    for all ``i`` in {2,nparticles}

    Args:
        xs (Tensor): Random number input with shape=(b, nparticles - 2)

    Returns:
        u_i (Tensor): solution with shape=(b, nparticles - 2)
    """
    return RootFinderPolynomial.apply(xs)


def get_xi_parameter(p0: Tensor, mass: Tensor) -> Tensor:
    """Returns the solution of the equation
    ...math::
        e_cm = sum_i sqrt(m_i^2 + xi^2 * p0_i^2) for xi

    Args:
        p0 (Tensor): energies with shape=(b, nparticles)
        m (Tensor): particle masses with shape=(1, nparticles)

    Returns:
        xi (Tensor): scaling parameter with shape=(b,)
    """
    e_cm = p0.sum(dim=1)
    func = lambda x: func_mass(x, p0, mass, e_cm)
    dxif = lambda x: dxifunc_mass(x, p0, mass)
    guess = 0.5 * torch.ones((p0.shape[0],))
    xi = newton(func, dxif, 0.0, 1.0, guess)
    return xi
    # return RootFinderMass.apply(p0, mass)
