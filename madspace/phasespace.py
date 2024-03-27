from typing import Optional
import numpy as np
import torch

from .propagators import massless_propogator, unstable_massive_propogator
from .base import PhaseSpaceMapping
from .helper import kaellen, rotate_zy, boost, lsquare
from .vectors import LorentzVector


class GenDecay(PhaseSpaceMapping):
    def __init__(
        self,
        m1: torch.Tensor,
        m2: torch.Tensor,
    ):
        super().__init__(dims_in=2, invertible=False)
        self.m1 = m1  # Mass of decay particle 1
        self.m2 = m2  # Mass of decay particle 2
        self.pi = torch.tensor(np.pi)

    def map(self, r: torch.Tensor, p0: torch.Tensor):
        """Forward pass from random numbers to momenta

        Args:
            r (torch.Tensor): 2-dimensional random number input with shape=(n, 2)
            p0 (torch.Tensor): momenta of decaying particle (in lab frame) with shape=(n, 4)

        Returns:
            torch.Tensor: decay momenta p1 and p2 in (lab frame)
        """
        p1 = torch.zeros(r.shape[0],4, device=r.device)
        p2 = torch.zeros(r.shape[0],4, device=r.device)
        s = lsquare(p0)
        r1, r2 = r[:, 0], r[:, 1]

        # Define the angles
        phi = 2 * self.pi * r1
        theta = 2 * r2 - 1

        # Define the momenta (in COM frame of decaying particle)
        p1[:, 0] = (s + self.m1**2 - self.m2**2) / (2 * torch.sqrt(s))
        p1[:, 3] = torch.sqrt(kaellen(s, self.m1**2, self.m2**2)) / (2 * torch.sqrt(s))
        
        # First rotate, then boost into lab-frame
        p1 = rotate_zy(p1, phi, theta)
        p1 = boost(p1, p0)
        p2 = p0 - p1
        
        # TODO: Check/Think if this is correct or if 1/det is needed
        det = torch.sqrt(kaellen(s, self.m1**2, self.m2**2)) * self.pi / (2 * s) 

        return torch.cat((p1, p2), 1), det
    
    def inverse_map(self, p: torch.Tensor):
        raise NotImplementedError
