from typing import Optional
import numpy as np
import torch

from .base import Mapping
from .functional.phasespace.propagators import massless_propogator, unstable_massive_propogator

class TwoParticlePhasespaceA(Mapping):

    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        sqrt_s_min: float = 50.,
        e_beam: float = 6500.,
        s_mass: float = 0.,
        s_gamma: float = 0.,
        nu: float = 0.95
    ):
        super().__init__(dims_in, dims_c)

        self.register_buffer("e_beam", torch.tensor(e_beam))
        self.register_buffer("s_min", torch.tensor(sqrt_s_min**2))
        self.register_buffer("s_max", torch.tensor(4*e_beam**2))
        self.register_buffer("pi", torch.tensor(np.pi))

        self.massless = s_mass == 0.
        if self.massless:
            self.nu = nu
        else:
            self.y1 = torch.atan((self.s_min - s_mass**2) / (s_mass*s_gamma))
            self.y2 = torch.atan((self.s_max - s_mass**2) / (s_mass*s_gamma))
            self.s_mass = torch.tensor(s_mass)
            self.s_gamma = torch.tensor(s_gamma)

    def _sublogdet(self, s, r2, r3, r4):
        del r4
        return torch.log(
            self.pi * (s/self.s_max)**(-2 * r2) *
            (-r3*s + (-1 + r3) * (s/self.s_max)**(2 * r2) * self.s_max) *
            (s - r3 * s + r3 * (s/self.s_max)**(2 * r2) * self.s_max) *
            torch.log(s/self.s_max) / (4 * self.s_max)
        )

    def _forward(self, p, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        px1, py1, pz1, pz2 = torch.unbind(p, dim=-1)
        e1 = torch.sqrt(px1**2 + py1**2 + pz1**2)
        e2 = torch.sqrt(px1**2 + py1**2 + pz2**2)
        pz_tot = pz1 + pz2
        e_tot = e1 + e2
        x1 = (e_tot + pz_tot) / (2*self.e_beam)
        x2 = (e_tot - pz_tot) / (2*self.e_beam)
        s = self.s_max * x1 * x2
        r2 = torch.log(x1) / torch.log(s / self.s_max)
        r3 = (pz1/self.e_beam + x2) / (x1 + x2)
        r4 = torch.atan2(py1, px1) / (2*self.pi) + 0.5

        if self.massless:
            r1 = (
                (s**(1-self.nu) - self.s_min**(1-self.nu))
                / (self.s_max**(1-self.nu) - self.s_min**(1-self.nu))
            )
            logdet = - torch.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            r1 = (
                (torch.atan((s - self.s_mass**2) / (self.s_mass * self.s_gamma)) - self.y1)
                / (self.y2 - self.y1)
            )
            logdet = - torch.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )

        r = torch.stack((r1, r2, r3, r4), dim=-1)
        return r, -logdet - self._sublogdet(s, r2, r3, r4)

    def _inverse(self, r, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = torch.unbind(r, dim=-1)
        # Mapping of s as defined in https://arxiv.org/pdf/hep-ph/0206070.pdf (p. 17)
        if self.massless:
            s = (
                r1 * self.s_max**(1-self.nu) +
                (1 - r1) * self.s_min**(1-self.nu)
            ) ** (1 / (1-self.nu))
            logdet = - torch.log(
                (1 - self.nu) /
                (s**self.nu * (self.s_max**(1-self.nu) - self.s_min**(1-self.nu)))
            )
        else:
            s = (
                self.s_mass * self.s_gamma * torch.tan(self.y1 + (self.y2 - self.y1)*r1)
                + self.s_mass**2
            )
            logdet = - torch.log(
                self.s_mass * self.s_gamma / (
                    (self.y2 - self.y1) *
                    ((s - self.s_mass**2)**2 + self.s_mass**2 * self.s_gamma**2)
                )
            )
        x1 = (s / self.s_max)**r2
        x2 = (s / self.s_max)**(1-r2)
        pz1 = self.e_beam * (x1*r3 + x2*(r3-1))
        pz2 = self.e_beam * (x1*(1-r3) - x2*r3)
        pt = torch.sqrt(s*r3*(1-r3))
        phi = 2 * self.pi * (r4 - 0.5)
        px1 = pt * torch.cos(phi)
        py1 = pt * torch.sin(phi)
        p = torch.stack((px1, py1, pz1, pz2), dim=-1)
        return p, logdet + self._sublogdet(s, r2, r3, r4)

    def _log_det(self, x_or_z, condition, inverse=False, **kwargs):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition, **kwargs)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition, **kwargs)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample


class TwoParticlePhasespaceB(Mapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        sqrt_s_min: float = 50.,
        e_beam: float = 6500.,
        s_mass: float = 0.,
        s_gamma: float = 0.,
        nu: float = 0.95
    ):
        super().__init__(dims_in, dims_c)

        self.e_beam = torch.tensor(e_beam)
        self.s_min = torch.tensor(sqrt_s_min**2)
        self.s_max = torch.tensor(4*e_beam**2)
        self.pi = torch.tensor(np.pi)

        self.massless = s_mass == 0.
        if self.massless:
            self.nu = nu
        else:
            self.y1 = torch.atan((self.s_min - s_mass**2) / (s_mass*s_gamma))
            self.y2 = torch.atan((self.s_max - s_mass**2) / (s_mass*s_gamma))
            self.s_mass = torch.tensor(s_mass)
            self.s_gamma = torch.tensor(s_gamma)

    def _sublogdet(self, s):
        det = 4 * self.pi * torch.log(self.s_max/s) / self.s_max
        return torch.log(det)

    def _forward(self, p, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        x1, x2, costheta, phi = torch.unbind(p, dim=-1)
        s = self.s_max * x1 * x2
        r2 = torch.log(x1) / torch.log(s / self.s_max)
        r3 = (costheta + 1)/2
        r4 = phi / (2*self.pi) + 0.5

        # Mapping of s (see functional.phasespace.propagators for details)
        if self.massless:
            r1, logdet = massless_propogator(s, self.s_min, self.s_max, self.nu, inverse=False)
        else:
            r1, logdet = unstable_massive_propogator(
                s, self.s_min, self.s_max, self.s_mass, self.s_gamma, inverse=False
            )

        r = torch.stack((r1, r2, r3, r4), dim=-1)
        return r, logdet - self._sublogdet(s)

    def _inverse(self, r, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = torch.unstack(r, dim=-1)
        # Mapping of s (see functional.phasespace.propagators for details)
        if self.massless:
            s, logdet = massless_propogator(r1, self.s_min, self.s_max, self.nu, inverse=True)
        else:
            s, logdet = unstable_massive_propogator(
                r1, self.s_min, self.s_max, self.s_mass, self.s_gamma, inverse=True
            )
        
        x1 = (s / self.s_max)**r2
        x2 = (s / self.s_max)**(1-r2)
        costheta = 2 * r3 - 1
        phi = 2 * self.pi * (r4 - 0.5)

        p = torch.stack((x1, x2, costheta, phi), dim=-1)
        return p, logdet + self._sublogdet(s)

    def _log_det(self, x_or_z, condition, inverse=False, **kwargs):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition, **kwargs)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition, **kwargs)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample


class TwoParticlePhasespaceFlatB(Mapping):

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        super().__init__(dims_in, dims_c)
        self.pi = torch.tensor(np.pi)

    def _forward(self, p, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        x1, x2, costheta, phi = torch.unbind(p, dim=-1)

        r1 = x1
        r2 = x2
        r3 = (costheta + 1)/2
        r4 = phi / (2*self.pi) + 0.5

        logdet = torch.log(4 * self.pi)
        r = torch.stack((r1, r2, r3, r4), dim=-1)
        return r, -logdet

    def _inverse(self, r, condition, **kwargs):
        # Note: the condition is ignored.
        del condition

        r1, r2, r3, r4 = torch.unbind(r, dim=-1)

        x1 = r1
        x2 = r2
        costheta = 2 * r3 - 1
        phi = 2 * self.pi * (r4 - 0.5)

        logdet = torch.log(4 * self.pi)
        p = torch.stack((x1, x2, costheta, phi), dim=-1)
        return p, logdet

    def _log_det(self, x_or_z, condition, inverse=False, **kwargs):
        if inverse:
            # the log-det of the inverse function (log dF^{-1}/dz)
            _, logdet = self._inverse(x_or_z, condition, **kwargs)
            return logdet
        else:
            # the log-det of the forward pass (log dF/dx)
            _, logdet = self._forward(x_or_z, condition, **kwargs)
            return logdet

    def _sample(self, num_samples, condition):
        r_values = self.base_dist.sample(num_samples, condition)
        sample, _ = self.inverse(r_values, condition)
        return sample
