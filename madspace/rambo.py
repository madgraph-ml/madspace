import numpy as np
from typing import Optional, Tuple
from math import gamma, pi
import torch
from torch import Tensor, cos, sin, cosh, sinh, sqrt, log
import torch.functional as F

from .rootfinder.roots import get_u_parameter, get_xi_parameter


from .base import PhaseSpaceMapping
from .helper import (
    MINKOWSKI,
    map_fourvector_rambo,
    two_body_decay_factor,
    boost,
    boost_beam,
    lsquare,
)


class Rambo(PhaseSpaceMapping):
    """Rambo algorithm as presented in
    [1] Rambo [Comput. Phys. Commun. 40 (1986) 359-373]


    """

    def __init__(
        self,
        e_cm: float,
        nparticles: int,
        masses: list[float] = None,
    ):
        self.n_particles = nparticles
        self.e_cm = e_cm

        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.n_particles
            e_min = sum(masses)
        else:
            self.masses = masses
            e_min = 0.0

        if not self.e_cm >= e_min:
            raise ValueError(f"COM energy needs to be larger than sum of external")

        dims_in = [(nparticles, 4)]
        dims_out = [(nparticles, 4)]
        super().__init__(dims_in, dims_out)

    def map(self, inputs, condition=None):
        del condition
        r = inputs[0]  # have dims (b,n,4)

        # Construct intermediate particle momenta
        q = map_fourvector_rambo(r)

        # sum over all particles
        Q = q.sum(dim=1, keepdim=True)  # has shape (b,1,4)

        # Get scaling factor and match dimensions
        M = sqrt(lsquare(Q))  # has shape (b,1)
        M2 = M[..., None]
        x = self.e_cm / M2  # has shape (b,1,1)

        # Boost and refactor
        p = boost(q, Q, inverse=True)
        p = x * p

        w0 = torch.ones((r.shape[0],))
        gs = w0 * self._massles_weight()

        if self.masses is not None:
            # match dimensions of masses
            m = self.masses[None, ...]

            # solve for xi in massive case, see Ref. [1]
            xi = get_xi_parameter(p[:, :, 0], m)

            # Make momenta massive
            xi = xi[:, None, None]
            k = torch.ones_like(p)
            k[:, :, 0] = torch.sqrt(m**2 + xi[:, :, 0] ** 2 * p[:, :, 0] ** 2)
            k[:, :, 1:] = xi * p[:, :, 1:]

            # Get massive density
            w_m = self._massive_weight(k, p, xi[:, 0, 0])

            return (k,), w_m * gs

        return (p,), gs

    def map_inverse(self, inputs, condition=None):
        """Does not exist for Rambo"""
        raise NotImplementedError

    def _massles_weight(self):
        w0 = (
            (pi / 2.0) ** (self.n_particles - 1)
            * self.e_cm ** (2 * self.n_particles - 4)
            / (gamma(self.n_particles) * gamma(self.n_particles - 1))
        )
        return w0

    def _massive_weight(
        self,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            k (torch.Tensor): massive momenta in shape=(b,n,4)
            p (torch.Tensor): massless momenta in shape=(b,n,4)
            xi (torch.Tensor, Optional): shift variable with shape=(b,)

        Returns:
            torch.Tensor: massive weight
        """
        # get correction factor for massive ones
        ks2 = k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2
        ps2 = p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2
        k0 = k[:, :, 0]
        p0 = p[:, :, 0]
        w_M = (
            xi ** (3 * self.n_particles - 3)
            * torch.prod(p0 / k0, dim=1)
            * torch.sum(ps2 / p0, dim=1)
            / torch.sum(ks2 / k0, dim=1)
        )
        return w_M

    def density(self, inputs, condition=None, inverse=False):
        del condition
        if inverse:
            raise NotImplementedError

        _, gs = self.map(self, inputs)
        return gs
