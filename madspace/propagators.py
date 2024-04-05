""" 
Implement basic building blocks.
Based on the mappings described in
    [1] https://arxiv.org/abs/hep-ph/0206070v2
    and described more precisely in
    [2] https://arxiv.org/abs/hep-ph/0008033
    [3] https://freidok.uni-freiburg.de/data/154629
"""

from typing import Optional
import torch
from torch import Tensor, sqrt, log, atan2, atan, tan

from .base import PhaseSpaceMapping, TensorList
from .helper import kaellen, rotate_zy, boost, lsquare, edot, pi


class BreitWignerPropagator(PhaseSpaceMapping):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        mass: Tensor,
        width: Tensor,
    ):
        super().__init__(dims_r=(1,), dims_p=(1,))

        self.s_min = s_min
        self.s_max = s_max

        self.m2 = mass**2
        self.gm = mass * width

        self.y1 = atan((self.s_min - self.m2) / (self.gm))
        self.y2 = atan((self.s_max - self.m2) / (self.gm))
        self.dy21 = self.y2 - self.y1

    def _inverse(self, r: Tensor, condition: Tensor):
        """Inverse pass from random numbers to momenta

        Args:
            r (Tensor): 1-dimensional random number input with shape=(b,1)

        Returns:
            s (Tensor): invariant s with shape=(b,1)
        """
        del condition
        s = self.gm * tan(self.y1 + self.dy12 * r) + self.m2
        gs = self.gm / (self.dy12 * ((s - self.m2) ** 2 + self.gm**2))

        return s, -log(gs)

    def _forward(self, s: Tensor, condition: Tensor):
        """Forward pass from invariant s onto random number r

        Args:
            s (Tensor): invariant s with shape=(b,1)

        Returns:
            r (Tensor): 1-dimensional random number with shape=(b,1)
        """
        del condition
        r = (atan((s - self.m2) / (self.gm)) - self.y1) / self.dy12
        gs = self.gm / (self.dy12 * ((s - self.m2) ** 2 + self.gm**2))

        return r, log(gs)

    def _log_det(
        self,
        s_or_r: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ):
        del condition
        s = self.gm * tan(self.y1 + self.dy12 * s_or_r) + self.m2 if inverse else s_or_r
        gs = self.gm / (self.dy12 * ((s - self.m2) ** 2 + self.gm**2))
        return (-1) ** inverse * log(gs)


class ZeroWidthPropagator(PhaseSpaceMapping):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        nu: float = 1.4,
        mass: Tensor = None,
    ):
        super().__init__(dims_r=(1,), dims_p=(1,))

        self.s_min = s_min
        self.s_max = s_max
        self.nu = nu
        self.power = 1 - self.nu

        if mass is None:
            self.m2 = -1e-6
        else:
            self.m2 = mass**2

        self.q_max = self.s_max - self.m2
        self.q_min = self.s_min - self.m2

        if self.nu == 1.0:
            self._forward = self.forward_1
            self._inverse = self.inverse_1
            self._log_det = self.log_det_1
        else:
            self._forward = self.forward_nu
            self._inverse = self.inverse_nu
            self._log_det = self.log_det_nu

    # ---------------------------------------#
    # Mappings for nu = 1

    def inverse_1(self, r: Tensor, condition: Tensor):
        """Inverse pass from random number r to invariant s"""
        del condition
        s = self.q_max**r * self.q_min ** (1 - r) + self.m2
        gs = 1 / ((s - self.m2) * (log(self.q_max) - log(self.q_min)))

        return s, -log(gs)

    def forward_1(self, s: Tensor, condition: Tensor):
        """Forward pass from invariant s to random number r"""
        del condition
        r = torch.log((s - self.m2) / self.q_min) / torch.log(self.q_max / self.q_min)
        gs = 1 / ((s - self.m2) * (log(self.q_max) - log(self.q_min)))

        return r, log(gs)

    def log_det_1(
        self,
        s_or_r: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ):
        del condition
        s = (
            self.q_max**s_or_r * self.q_min ** (1 - s_or_r) + self.m2
            if inverse
            else s_or_r
        )
        gs = 1 / ((s - self.m2) * (log(self.q_max) - log(self.q_min)))
        return (-1) ** inverse * log(gs)

    # ---------------------------------------#
    # Mappings for nu != 1

    def inverse_nu(self, r: Tensor, condition: Tensor):
        """Inverse pass from random number r to invariant s"""
        del condition
        qmaxpow = self.q_max**self.power
        qminpow = self.q_min**self.power
        s = (r * qmaxpow + (1 - r) * qminpow) ** (1 / self.power) + self.m2
        gs = self.power / ((qmaxpow - qminpow) * (s - self.m2) ** self.nu)

        return s, -log(gs)

    def forward_nu(self, s: Tensor, condition: Tensor):
        """Forward pass from invariant s to random number r"""
        del condition
        qmaxpow = self.q_max**self.power
        qminpow = self.q_min**self.power
        spow = (s - self.m2) ** self.power
        r = (spow - qminpow) / (qmaxpow - qminpow)
        gs = self.power / ((qmaxpow - qminpow) * (s - self.m2) ** self.nu)

        return r, log(gs)

    def log_det_nu(
        self,
        s_or_r: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ):
        del condition
        qmaxpow = self.q_max**self.power
        qminpow = self.q_min**self.power
        s = (
            (s_or_r * qmaxpow + (1 - s_or_r) * qminpow) ** (1 / self.power) + self.m2
            if inverse
            else s_or_r
        )
        gs = self.power / ((qmaxpow - qminpow) * (s - self.m2) ** self.nu)
        return (-1) ** inverse * log(gs)


class MasslessPropagator(ZeroWidthPropagator):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        nu: float = 1.4,
    ):
        super().__init__(s_min, s_max, nu=nu, mass=None)
        

