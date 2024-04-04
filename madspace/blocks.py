""" 
Implement basic building blocks.
Based on the mappings described in
    [1] https://arxiv.org/abs/hep-ph/0206070v2
    and described more precisely in
    [2] https://arxiv.org/abs/hep-ph/0008033
    [3] https://freidok.uni-freiburg.de/data/154629
"""

import torch
from torch import Tensor, sqrt, log, atan2, atan, tan

from .base import PhaseSpaceMapping
from .helper import kaellen, rotate_zy, boost, lsquare, edot, pi


class DecayBlock(PhaseSpaceMapping):
    """
    Implement isotropic decay, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
    ):
        super().__init__(dims_r=(2,), dims_p=(2,4), dims_c=(1,4))
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2
        self.pi = torch.tensor(pi)

    def _density(self, p: Tensor) -> Tensor:
        """Calculates the associated phase-space density
        according to Eq. (C.8) - (C.10) in [1]

        Args:s
            p (Tensor): Total momentum of the decay process with shape=(b,4)

        Returns:
            g (Tensor): returns the density with shape=(b,)
        """
        s = lsquare(p)
        g = (2 * s) / sqrt(kaellen(s, self.m1**2, self.m2**2)) / self.pi
        return g

    def _inverse(self, r: Tensor, cp: Tensor):
        """Inverse pass from random numbers to momenta

        Args:
            r (Tensor): random numbers with shape=(b,2)
            cp (Tensor): input momentum (lab frame) with shape=(b,1,4)

        Returns:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
        """
        p0 = cp[:,0]
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        s = lsquare(p0)
        if torch.any(s < 0):
            raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * self.pi * r1
        costheta = 2 * r2 - 1

        # Define the momenta (in COM frame of decaying particle)
        p1[:, 0] = (s + self.m1**2 - self.m2**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, self.m1**2, self.m2**2)) / (2 * sqrt(s))

        # First rotate, then boost into lab-frame
        p1 = rotate_zy(p1, phi, costheta)
        p1 = boost(p1, p0)
        p2 = p0 - p1

        # get the log determinant
        logdet = log(self._density(p0))

        return torch.stack([p1, p2], dim=1), -logdet

    def _forward(self, p_decay: Tensor, cp: Tensor):
        """Forward pass from decay momenta onto random numbers

        Args:
            p_decay (Tensor): decay momenta (lab frame) with shape=(b,2,4)
            cp (Tensor): input momentum (lab frame) with shape=(b,1,4)

        Returns:
            r (Tensor): random numbers with shape=(b,2)
        """
        del condition
        # Decaying particle in lab-frame
        p0 = p_decay.sum(dim=1)
        p1 = p_decay[:, 0]

        # Boost p1 into COM
        p1 = boost(p1, -p0)

        # get p1 absolute momentum
        p1mag = sqrt(edot(p1[:, 1:], p1[:, 1:]))

        # Extract phi and theta
        costheta = p1[:, 3] / p1mag
        phi = atan2(p1[:, 2], p1[:, 1])

        # Get the random numbers
        r1 = phi / 2 / self.pi
        r2 = (costheta + 1) / 2

        # get the log determinant
        logdet = log(self._density(p0))

        return torch.stack([r1, r2], dim=1), logdet

    def _log_det(
        self,
        p_or_r: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ):
        p0 = condition
        logdet = log(self._density(p0))
        return (-1) ** (inverse) * logdet
    
class tChannelBlock(PhaseSpaceMapping):
    """
    Implement anisotropic decay, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
        [2] https://freidok.uni-freiburg.de/data/154629
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
    ):
        super().__init__(dims_in=2, dims_c=1)
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2
        self.pi = torch.tensor(pi)

    def _inverse(self, r: Tensor, condition: Tensor):
        """Inverse pass from random numbers to momenta

        Args:
            r (Tensor): 2-dimensional random number input with shape=(b,2)
            condition (Tensor): momenta of decaying particle (in lab frame) with shape=(b,1,4)

        Returns:
            Tensor: decay momenta p1 and p2 in (lab frame) with shape=(b,2,4)
        """
        p0 = condition
        p1 = torch.zeros(r.shape[0], 4, device=r.device)
        p2 = torch.zeros(r.shape[0], 4, device=r.device)
        s = lsquare(p0)
        if torch.any(s < 0):
            raise ValueError(f"s needs to be always positive")

        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * self.pi * r1
        costheta = 2 * r2 - 1

        # Define the momenta (in COM frame of decaying particle)
        p1[:, 0] = (s + self.m1**2 - self.m2**2) / (2 * sqrt(s))
        p1[:, 3] = sqrt(kaellen(s, self.m1**2, self.m2**2)) / (2 * sqrt(s))

        # First rotate, then boost into lab-frame
        p1 = rotate_zy(p1, phi, costheta)
        p1 = boost(p1, p0)
        p2 = p0 - p1

        # get the log determinant
        logdet = log(self._density(p0))

        return torch.stack([p1, p2], dim=1), -logdet


class sChannelBWBlock(PhaseSpaceMapping):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        mass: Tensor,
        width: Tensor,
    ):
        super().__init__(dims_in=1, dims_c=None)

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


class sChannelZeroWidthBlock(PhaseSpaceMapping):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        nu: float = 1.4,
        mass: Tensor = None,
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


class sChannelMasslessBlock(sChannelZeroWidthBlock):
    def __init__(
        self,
        s_min: Tensor,
        s_max: Tensor,
        nu: float = 1.4,
    ):
        super().__init__(s_min, s_max, nu=nu, mass=None)
