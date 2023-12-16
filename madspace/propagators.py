""" Implement propagator mappings.
    Based on the mappings described in
    [1] https://arxiv.org/abs/hep-ph/0206070v2
    and described more precisely in
    [2] https://arxiv.org/abs/hep-ph/0008033
    [3] https://freidok.uni-freiburg.de/data/154629"""

import torch
import torch.nn as nn
from .base import PhaseSpaceGenerator


class UnstableMassivePropagator(PhaseSpaceGenerator):
    def __init__(
        self,
        s_min: torch.Tensor,
        s_max: torch.Tensor,
        mass: torch.Tensor,
        width: torch.Tensor,
    ):
        super().__init__(dims_in=1, dims_c=None)

        self.s_min = s_min
        self.s_max = s_max
        self.mass = mass
        self.width = width

        self.y1 = torch.atan((self.s_min - self.mass**2) / (self.mass * self.width))
        self.y2 = torch.atan((self.s_max - self.mass**2) / (self.mass * self.width))

    def forward(self, s: torch.Tensor):
        """Forward pass from invariant s to random number r"""
        r = (torch.atan((s - self.mass**2) / (self.mass * self.width)) - self.y1) / (
            self.y2 - self.y1
        )
        det = (
            self.mass
            * self.width
            / (
                (self.y2 - self.y1)
                * ((s - self.mass**2) ** 2 + self.mass**2 * self.width**2)
            )
        )

        return r, det

    def inverse(self, r: torch.Tensor):
        """Inverse pass from random number r to invariant s"""
        s = (
            self.mass * self.width * torch.tan(self.y1 + (self.y2 - self.y1) * r)
            + self.mass**2
        )
        det = (
            (self.y2 - self.y1)
            * ((s - self.mass**2) ** 2 + self.mass**2 * self.width**2)
        ) / (self.mass * self.width)

        return s, det


# UNTIL HERE
## TODO : Do the rest later


def stable_massive_propogator(
    r_or_s: torch.Tensor,
    s_min: torch.Tensor,
    s_max: torch.Tensor,
    mass: torch.Tensor,
    nu: float = 0.95,
    inverse: bool = True,
):
    # Energy needs to be higher than mass
    assert s_min > mass**2
    if nu == 1.0:
        if inverse:
            s = (
                torch.exp(
                    r_or_s * torch.log(s_max - mass**2)
                    + (1 - r_or_s) * torch.log(s_min - mass**2)
                )
                + mass**2
            )
            logdet = -torch.log(
                (torch.log(s_max - mass**2) - torch.log(s_min - mass**2))
                * (s - mass**2)
            )
            return s, -logdet
        else:
            r = (torch.log(r_or_s - mass**2) - torch.log(s_min - mass**2)) / (
                torch.log(s_max - mass**2) - torch.log(s_min - mass**2)
            )
            logdet = -torch.log(
                (torch.log(s_max - mass**2) - torch.log(s_min - mass**2))
                * (r_or_s - mass**2)
            )
            return r, logdet
    else:
        if inverse:
            s = (
                r_or_s * (s_max - mass**2) ** (1 - nu)
                + (1 - r_or_s) * (s_min - mass**2) ** (1 - nu)
            ) ** (1 / (1 - nu)) + mass**2
            logdet = torch.log(
                (1 - nu)
                / (
                    (s - mass**2) ** nu
                    * (
                        (s_max - mass**2) ** (1 - nu)
                        - (s_min - mass**2) ** (1 - nu)
                    )
                )
            )
            return s, -logdet
        else:
            r = ((r_or_s - mass**2) ** (1 - nu) - (s_min - mass**2) ** (1 - nu)) / (
                (s_max - mass**2) ** (1 - nu) - (s_min - mass**2) ** (1 - nu)
            )
            logdet = torch.log(
                (1 - nu)
                / (
                    (r_or_s - mass**2) ** nu
                    * (
                        (s_max - mass**2) ** (1 - nu)
                        - (s_min - mass**2) ** (1 - nu)
                    )
                )
            )
            return r, logdet


def massless_propogator(
    r_or_s: torch.Tensor,
    s_min: torch.Tensor,
    s_max: torch.Tensor,
    nu: float = 0.95,
    m2_eps: float = -1e-8,
    inverse: bool = True,
):
    if nu == 1.0:
        if inverse:
            s = (
                torch.exp(
                    r_or_s * torch.log(s_max - m2_eps)
                    + (1 - r_or_s) * torch.log(s_min - m2_eps)
                )
                + m2_eps
            )
            logdet = -torch.log(
                (torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps)) * (s - m2_eps)
            )
            return s, -logdet
        else:
            r = (torch.log(r_or_s - m2_eps) - torch.log(s_min - m2_eps)) / (
                torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps)
            )
            logdet = -torch.log(
                (torch.log(s_max - m2_eps) - torch.log(s_min - m2_eps))
                * (r_or_s - m2_eps)
            )
            return r, logdet
    if inverse:
        s = (
            r_or_s * (s_max - m2_eps) ** (1 - nu)
            + (1 - r_or_s) * (s_min - m2_eps) ** (1 - nu)
        ) ** (1 / (1 - nu)) + m2_eps
        logdet = torch.log(
            (1 - nu)
            / (
                (s - m2_eps) ** nu
                * ((s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu))
            )
        )
        return s, -logdet
    else:
        r = ((r_or_s - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu)) / (
            (s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu)
        )
        logdet = torch.log(
            (1 - nu)
            / (
                (r_or_s - m2_eps) ** nu
                * ((s_max - m2_eps) ** (1 - nu) - (s_min - m2_eps) ** (1 - nu))
            )
        )
        return r, logdet
