import torch
from torch import Tensor, sqrt, log, atan2

from .base import Mapping
from .helper import kaellen, rotate_zy, boost, lsquare, edot, pi


class DecayBlock(Mapping):
    """
    Implement isotropic decay, based on the mapping described in
        [1] https://arxiv.org/abs/hep-ph/0008033
    """

    def __init__(
        self,
        m1: Tensor,
        m2: Tensor,
    ):
        super().__init__(dims_in=2)
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2
        self.pi = torch.tensor(pi)

    def _density(self, p: Tensor) -> Tensor:
        """Calculates the associated phase-space density
        according to Eq. (C.8) - (C.10) in [1]

        Args:s
            p (Tensor): Total momentum of the decay process with shape=(b,4)

        Returns:
            g (Tensor): returns the density with shape=(b,4)
        """
        s = lsquare(p)
        g = (2 * s) / sqrt(kaellen(s, self.m1**2, self.m2**2)) / self.pi
        return g

    def _inverse(self, r: Tensor, condition: Tensor):
        """Inverse pass from random numbers to momenta

        Args:
            r (Tensor): 2-dimensional random number input with shape=(b,2)
            condition (Tensor): momenta of decaying particle (in lab frame) with shape=(b,4)

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
        p1 = rotate_zy(p1, phi, torch.acos(costheta))
        p1 = boost(p1, p0)
        p2 = p0 - p1

        # get the log determinant
        logdet = log(self._density(p0))

        return torch.stack([p1, p2], dim=1), -logdet

    def _forward(self, p_decay: Tensor, condition: Tensor):
        """Forward pass from decay momenta onto random numbers

        Args:
            p_decay (Tensor): decay momenta (lab-frame) with shape=(b,2,4)
            condition (Tensor): momenta of decaying particle (in lab frame) with shape=(b,4)

        Returns:
            r (Tensor): random number with shape=(b,2,4)
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
        x_or_z: Tensor,
        condition: Tensor,
        inverse: bool = False,
    ):
        p0 = condition
        logdet = log(self._density(p0))
        return (-1) ** (inverse) * logdet
