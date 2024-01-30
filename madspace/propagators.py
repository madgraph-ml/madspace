""" Implement propagator mappings.
    Based on the mappings described in
    [1] https://arxiv.org/abs/hep-ph/0206070v2
    and described more precisely in
    [2] https://arxiv.org/abs/hep-ph/0008033
    [3] https://freidok.uni-freiburg.de/data/154629"""

import torch
import torch.nn as nn
from .base import PhaseSpaceGenerator


class UnstablePropagator(PhaseSpaceGenerator):
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

        self.m2 = mass**2
        self.gm = mass * width

        self.y1 = torch.atan((self.s_min - self.m2) / (self.gm))
        self.y2 = torch.atan((self.s_max - self.m2) / (self.gm))
        self.dy21 = self.y2 - self.y1

    def _forward(self, s: torch.Tensor, condition: torch.Tensor):
        """Forward pass from invariant s to random number r"""
        r = (torch.atan((s - self.m2) / (self.gm)) - self.y1) / self.dy12
        det = self.gm / self.dy12 / ((s - self.m2) ** 2 + self.gm**2)

        return r, det

    def _inverse(self, r: torch.Tensor, condition: torch.Tensor):
        """Inverse pass from random number r to invariant s"""
        s = self.gm * torch.tan(self.y1 + self.dy12 * r) + self.m2
        det = self.dy12 * ((s - self.m2) ** 2 + self.gm**2) / self.gm

        return s, det


class StablePropagator(PhaseSpaceGenerator):
    def __init__(
        self,
        s_min: torch.Tensor,
        s_max: torch.Tensor,
        nu: float = 1.4,
        mass: torch.Tensor = None,
    ):
        super().__init__(dims_in=1, dims_c=None)

        self.s_min = s_min
        self.s_max = s_max
        self.nu = nu
        self.power = 1 - self.nu

        if mass is None:
            self.m2 = -1e-6
        else:
            self.m2 = mass**2

        self.q_max = self.s_max - self.m2
        self.q_min = self.s_max - self.m2

        if self.nu == 1.0:
            self._forward = self.forward_1
            self._inverse = self.inverse_1
        else:
            self._forward = self.forward_nu
            self._inverse = self.inverse_nu

    def forward_1(self, s: torch.Tensor, condition: torch.Tensor):
        """Forward pass from invariant s to random number r"""

        r = torch.log((s - self.m2) / self.q_min) / torch.log(self.q_max / self.q_min)
        det = (s - self.m2) * torch.log(self.q_max / self.q_min)

        return r, 1 / det

    def forward_nu(self, s: torch.Tensor, condition: torch.Tensor):
        """Forward pass from invariant s to random number r"""
        qmaxpow = self.q_max**self.power
        qminpow = self.q_min**self.power
        spow = (s - self.m2) ** self.power
        r = (spow - qminpow) / (qmaxpow - qminpow)
        det = (qmaxpow - qminpow) * (s - self.m2) ** self.nu / self.power

        return r, 1 / det

    def inverse_1(self, r: torch.Tensor, condition: torch.Tensor):
        """Inverse pass from random number r to invariant s"""
        s = self.q_max**r * self.q_min ** (1 - r) + self.m2
        det = (s - self.m2) * torch.log(self.q_max / self.q_min)

        return s, det

    def inverse_nu(self, r: torch.Tensor, condition: torch.Tensor):
        """Inverse pass from random number r to invariant s"""
        qmaxpow = self.q_max**self.power
        qminpow = self.q_min**self.power
        s = (r * qmaxpow + (1 - r) * qminpow) ** (1 / self.power) + self.m2
        det = (qmaxpow - qminpow) * (s - self.m2) ** self.nu / self.power

        return s, det

# TODO: Need to think about it again
class _TrainableStablePropagator(PhaseSpaceGenerator):
    def __init__(
        self,
        s_min: torch.Tensor,
        s_max: torch.Tensor,
        nu: float = 1.4,
        mass: torch.Tensor = None,
    ):
        super().__init__(dims_in=1, dims_c=None)

        self.s_min = s_min
        self.s_max = s_max
        self.nu = nn.Parameter(torch.tensor(nu))
        self.power = 1 - self.nu

        if mass is None:
            self.m2 = -1e-6
        else:
            self.m2 = mass**2

        self.q_max = self.s_max - self.m2
        self.q_min = self.s_max - self.m2
        self.eps = 1e-6

    def forward_1(self, s: torch.Tensor, condition: torch.Tensor):
        """Forward pass from invariant s to random number r"""

        r = torch.log((s - self.m2) / self.q_min) / torch.log(self.q_max / self.q_min)
        det = (s - self.m2) * torch.log(self.q_max / self.q_min)

        return r, 1 / det

    def forward_nu(self, s: torch.Tensor, condition: torch.Tensor):
        """Forward pass from invariant s to random number r"""
        qmaxpow = self.q_max ** (1 - self.nu)
        qminpow = self.q_min ** (1 - self.nu)
        spow = (s - self.m2) ** self.power
        r = (spow - qminpow) / (qmaxpow - qminpow)
        det = (qmaxpow - qminpow) * (s - self.m2) ** self.nu / self.power

        return r, 1 / det

    def _inverse(self, r: torch.Tensor, condition: torch.Tensor):
        """Inverse pass from random number r to invariant s"""

        if torch.lt(torch.abs(self.nu - 1), self.eps):
            s = self.q_max**r * self.q_min ** (1 - r) + self.m2
            det = (s - self.m2) * torch.log(self.q_max / self.q_min)
            if self.nu < 1:
                self.nu -= self.eps
            if self.nu > 1:
                self.nu += self.eps
        
        qmaxpow = self.q_max ** (1 - self.nu)
        qminpow = self.q_min ** (1 - self.nu)
        s = (r * qmaxpow + (1 - r) * qminpow) ** (1 / (1 - self.nu)) + self.m2
        det = (qmaxpow - qminpow) * (s - self.m2) ** self.nu / (1 - self.nu)

        return s, det
