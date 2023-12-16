""" Implement two-particle mappings.
    Bases on the mappings described in
    https://freidok.uni-freiburg.de/data/154629"""

import torch
import math as m
from .helper import kaellen

# TODO: Continue...
def two_particle_phasespace(
    r: torch.Tensor,
    s: torch.Tensor,
    m1: torch.Tensor = 0.,
    m2: torch.Tensor = 0.,
):
    """Two particle phase space
    parametrized in terms of the solid angle, i.e
        dPhi_{2} = V_{2}/(4 Pi) dcostheta dphi,
                 = V_{2} dr1 dr2,
    where V_{2} is the two-particle phase-space volume, given by
        V_{2} = 1/(8 Pi) * \lambda(s, m1^2, m2^2)^(1/2) / s
              = 1/(8 Pi), if [m1 = m2 = 0]
    with the Kaellen function `\lambda`.
    Args:
        r (torch.Tensor): random numbers input.
        s (torch.Tensor): squared CM energy.
        m1 (torch.Tensor, optional): mass of particle 1. Defaults to None.
        m2 (torch.Tensor, optional): mass of particle 2. Defaults to None.
    """

    # Define phase-space volume
    log_volume = (
        0.5 * torch.log(kaellen(s, m1**2, m2**2))
        - torch.log(s)
        - torch.log(8 * m.pi)
    )

    # do the mapping (linked to determinant)
    r1, r2 = torch.unbind(r, dim=-1)
    cos_theta = 2 * r1 - 1
    sin_theta = torch.sqrt(1 - cos_theta**2)
    phi = 2 * m.pi * (r2 - 0.5)
    logdet = torch.log(4 * m.pi)

    # parametrize the momenta in CM (not linked to determinant)
    p01 = (s + m1**2 - m2**2) / torch.sqrt(4 * s)
    p02 = (s - m1**2 + m2**2) / torch.sqrt(4 * s)
    pp = torch.sqrt(kaellen(s, m1**2, m2**2)) / torch.sqrt(4 * s)
    px1 = pp * sin_theta * torch.cos(phi)
    py1 = pp * sin_theta * torch.sin(phi)
    pz1 = pp * cos_theta

    p = torch.stack((p01, px1, py1, pz1, p02, -px1, -py1, -pz1), dim=-1)
    return p, log_volume - logdet
