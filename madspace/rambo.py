import numpy as np
from typing import Optional, Tuple
from scipy.optimize import brentq
from math import gamma, pi
import sys
import torch
import torch.functional as F

from .base import Mapping
from .helper import (
    MINKOWSKI,
    map_fourvector_rambo,
    two_body_decay_factor,
    boost,
    boost_beam,
    rambo_func,
    newton,
    mass_func,
)


class Rambo(Mapping):
    def __init__(self, e_cm, nparticles):
        self.e_cm = e_cm
        self.nparticles = nparticles
        super().__init__(nparticles * 4, nparticles * 4)

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = (
            (np.pi / 2.0) ** (nparticles - 1)
            * e_cm ** (2 * nparticles - 4)
            / (gamma(nparticles) * gamma(nparticles - 1))
        )
        return 1 / vol

    def map(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm

        p = np.empty((xs.shape[0], nparticles, 4))

        q = map_fourvector_rambo(xs.reshape(xs.shape[0], nparticles, 4))
        # sum over all particles
        Q = np.sum(q, axis=1)

        M = np.sqrt(np.einsum("kd,dd,kd->k", Q, MINKOWSKI, Q))
        b = -Q[:, 1:] / M[:, np.newaxis]
        x = e_cm / M
        gamma = Q[:, 0] / M
        a = 1.0 / (1.0 + gamma)

        bdotq = np.einsum("ki,kpi->kp", b, q[:, :, 1:])

        # make dimensions match
        gamma = gamma[:, np.newaxis]
        x = x[:, np.newaxis]
        p[:, :, 0] = x * (gamma * q[:, :, 0] + bdotq)

        # make dimensions match
        b = b[:, np.newaxis, :]  # dimensions: samples * nparticles * space dim)
        bdotq = bdotq[:, :, np.newaxis]
        x = x[:, :, np.newaxis]
        a = a[:, np.newaxis, np.newaxis]
        p[:, :, 1:] = x * (q[:, :, 1:] + b * q[:, :, 0, np.newaxis] + a * bdotq * b)

        return p.reshape(xs.shape)

    def pdf_gradient(self, xs):
        return 0

    def map_inverse(self, xs):
        raise NotImplementedError


class RamboOnDiet(PhaseSpaceMapping):
    def __init__(self, e_cm, nparticles):
        self.e_cm = np.float64(e_cm)  # The cast is important for accurate results!
        self.nparticles = nparticles
        super(RamboOnDiet, self).__init__(3 * nparticles - 4, 4 * nparticles)

    def map(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm

        p = np.empty((xs.shape[0], nparticles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = np.zeros((xs.shape[0], nparticles))
        u = np.empty((xs.shape[0], nparticles - 2))

        Q = np.tile([e_cm, 0, 0, 0], (xs.shape[0], 1))
        Q_prev = np.empty((xs.shape[0], 4))
        M[:, 0] = e_cm

        for i in range(2, nparticles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != nparticles:
                u[:, i - 2] = [
                    brentq(
                        lambda x: (
                            (nparticles + 1 - i) * x ** (2 * (nparticles - i))
                            - (nparticles - i) * x ** (2 * (nparticles + 1 - i))
                            - r_i
                        ),
                        0.0,
                        1.0,
                    )
                    for r_i in xs[:, i - 2]
                ]
                M[:, i - 1] = np.product(u[:, : i - 1], axis=1) * e_cm

            cos_theta = 2 * xs[:, nparticles - 6 + 2 * i] - 1
            phi = 2 * np.pi * xs[:, nparticles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2], M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * np.cos(phi) * np.sqrt(1 - cos_theta**2)
            p[:, i - 2, 2] = q * np.sin(phi) * np.sqrt(1 - cos_theta**2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = np.sqrt(q**2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2], MINKOWSKI)
            Q = boost(Q_prev, Q, MINKOWSKI)

        p[:, nparticles - 1] = Q

        return p.reshape((xs.shape[0], nparticles * 4))

    def map_inverse(self, p):
        count = p.size // (self.nparticles * 4)
        p = p.reshape((count, self.nparticles, 4))

        M = np.empty(p.shape[0])
        M_prev = np.empty(p.shape[0])
        Q = np.empty((p.shape[0], 4))
        r = np.empty((p.shape[0], 3 * self.nparticles - 4))

        Q[:] = p[:, -1]

        for i in range(self.nparticles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i - 2 :].sum(axis=1)
            M = np.sqrt(np.einsum("ij,jk,ik->i", P, MINKOWSKI, P))

            if i != self.nparticles:
                u = M_prev / M
                r[:, i - 2] = (self.nparticles + 1 - i) * u ** (
                    2 * (self.nparticles - i)
                ) - (self.nparticles - i) * u ** (2 * (self.nparticles + 1 - i))

            Q += p[:, i - 2]
            p_prime = boost(
                np.einsum("ij,ki->kj", MINKOWSKI, Q), p[:, i - 2], MINKOWSKI
            )
            r[:, self.nparticles - 6 + 2 * i] = 0.5 * (
                p_prime[:, 3] / np.sqrt(np.sum(p_prime[:, 1:] ** 2, axis=1)) + 1
            )
            phi = np.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.nparticles - 5 + 2 * i] = phi / (2 * np.pi) + (phi < 0)

        return r

    def jac(self, xs):
        return np.ones(xs.shape[0])

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = (
            (np.pi / 2.0) ** (nparticles - 1)
            * e_cm ** (2 * nparticles - 4)
            / (gamma(nparticles) * gamma(nparticles - 1))
        )
        return vol

    def pdf_gradient(self, xs):
        return 0


class RamboOnDietHadron(PhaseSpaceMapping):
    """Rambo on diet for Hadron colliders with masses"""

    def __init__(self, e_had: float, nparticles: int, masses: list = None):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            nparticles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super(RamboOnDietHadron, self).__init__(3 * nparticles - 2, 4 * nparticles)

        self.e_had = e_had
        self.nparticles = nparticles
        self.masses = masses

        # Make sure the list is as long as the number of
        # finals state particles if given
        if self.masses:
            assert len(self.masses) == self.nparticles

        # Define min energy due to masses (no cuts etc)
        e_min = np.sum(self.masses) if self.masses else 0
        self.tau_min = (e_min / self.e_had) ** 2

    def _get_parton_fractions(self, r):

        if self.tau_min > 0:
            logtau = r[0] * np.log(self.tau_min)
        else:
            logtau = np.log(r[0])

        logx1 = r[1] * logtau
        logx2 = (1 - r[1]) * logtau
        return np.exp(logx1), np.exp(logx2)

    def _get_pdf_random_numbers(self, x):
        tau = x[0] * x[1]
        r1 = np.log(tau) / np.log(self.tau_min)
        r2 = np.log(x[0]) / np.log(tau)
        return r1, r2

    def _get_rapidity_and_fractions(self, q):
        tau = (
            np.einsum("ij,jk,ik->i", q[:, 0, :], MINKOWSKI, q[:, 0, :])
            / self.e_had**2
        )[..., None]
        rapidity = np.arctanh(q[:, :, 3] / q[:, :, 0])
        logx1 = 0.5 * np.log(tau) + rapidity
        logx2 = 0.5 * np.log(tau) - rapidity
        return rapidity, np.exp(logx1), np.exp(logx2)

    def map(self, xs):
        xs, r1, r2 = xs[:, 2:], xs[:, [0]], xs[:, [1]]

        # get partonic energies and boost variables
        x1, x2 = self._get_parton_fractions([r1, r2])
        rapidity = 0.5 * np.log(x1 / x2)
        e_cm = self.e_had * np.sqrt(x1 * x2)

        p = np.empty((xs.shape[0], self.nparticles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = np.zeros((xs.shape[0], self.nparticles))
        u = np.empty((xs.shape[0], self.nparticles - 2))

        Q = e_cm * np.tile([1, 0, 0, 0], (xs.shape[0], 1))
        Q_prev = np.empty((xs.shape[0], 4))
        M[:, 0] = e_cm[:, 0]

        for i in range(2, self.nparticles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != self.nparticles:
                u[:, i - 2] = [
                    brentq(
                        lambda x: (
                            (self.nparticles + 1 - i) * x ** (2 * (self.nparticles - i))
                            - (self.nparticles - i)
                            * x ** (2 * (self.nparticles + 1 - i))
                            - r_i
                        ),
                        0.0,
                        1.0,
                    )
                    for r_i in xs[:, i - 2]
                ]
                M[:, i - 1] = np.product(u[:, : i - 1], axis=1) * e_cm[:, 0]

            cos_theta = 2 * xs[:, self.nparticles - 6 + 2 * i] - 1
            phi = 2 * np.pi * xs[:, self.nparticles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2], M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * np.cos(phi) * np.sqrt(1 - cos_theta**2)
            p[:, i - 2, 2] = q * np.sin(phi) * np.sqrt(1 - cos_theta**2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = np.sqrt(q**2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2], MINKOWSKI)
            Q = boost(Q_prev, Q, MINKOWSKI)

        p[:, self.nparticles - 1] = Q

        if self.masses:
            # Define masses
            m = np.tile(self.masses, (xs.shape[0], 1))

            # solve for massive case
            xi = np.empty((xs.shape[0], 1, 1))
            xi[:, 0, 0] = [
                brentq(
                    lambda x: (
                        np.sum(
                            np.sqrt(m[i, :] ** 2 + x**2 * p[i, :, 0] ** 2), axis=-1
                        )
                        - e_cm[i, 0]
                    ),
                    0.0,
                    1.0,
                )
                for i in range(xs.shape[0])
            ]

            # Make them massive
            k = np.empty((xs.shape[0], self.nparticles, 4))
            k[:, :, 0] = np.sqrt(m**2 + xi[:, :, 0] ** 2 * p[:, :, 0] ** 2)
            k[:, :, 1:] = xi * p[:, :, 1:]

            # Boost into hadronic lab frame
            k = boost_beam(k, rapidity, inverse=False)

            return k.reshape((xs.shape[0], self.nparticles * 4))

        # Boost into hadronic lab frame
        p = boost_beam(p, rapidity, inverse=False)

        return p.reshape((xs.shape[0], self.nparticles * 4))

    def map_inverse(self, k):
        count = k.size // (self.nparticles * 4)
        k = k.reshape((count, self.nparticles, 4))

        M = np.empty(k.shape[0])
        M_prev = np.empty(k.shape[0])
        Q = np.empty((k.shape[0], 4))
        r = np.empty((k.shape[0], 3 * self.nparticles - 4))

        # Boost into partonic CM frame and get x1 and x2
        q = np.sum(k, axis=1, keepdims=True)
        rapidity, x1, x2 = self._get_rapidity_and_fractions(q)
        e_cm = self.e_had * np.sqrt(x1 * x2)
        k = boost_beam(k, rapidity, inverse=True)

        # Make momenta massless
        p = np.empty((k.shape[0], self.nparticles, 4))
        if self.masses:
            # Define masses
            m = np.tile(self.masses, (k.shape[0], 1))

            # solve for mass case
            xi = np.empty((k.shape[0], 1, 1))
            xi[:, 0, 0] = [
                brentq(
                    lambda x: (
                        np.sum(np.sqrt(k[i, :, 0] ** 2 - m[i, :] ** 2), axis=-1)
                        - x * e_cm[i, 0]
                    ),
                    0.0,
                    1.0,
                )
                for i in range(k.shape[0])
            ]
            # Make them massive
            p[:, :, 0] = np.sqrt(k[:, :, 0] ** 2 - m**2) / xi[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:] / xi

        else:
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]

        Q[:] = p[:, -1]

        for i in range(self.nparticles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i - 2 :].sum(axis=1)
            M = np.sqrt(np.einsum("ij,jk,ik->i", P, MINKOWSKI, P))

            if i != self.nparticles:
                u = M_prev / M
                r[:, i - 2] = (self.nparticles + 1 - i) * u ** (
                    2 * (self.nparticles - i)
                ) - (self.nparticles - i) * u ** (2 * (self.nparticles + 1 - i))

            Q += p[:, i - 2]
            p_prime = boost(
                np.einsum("ij,ki->kj", MINKOWSKI, Q), p[:, i - 2], MINKOWSKI
            )
            r[:, self.nparticles - 6 + 2 * i] = 0.5 * (
                p_prime[:, 3] / np.sqrt(np.sum(p_prime[:, 1:] ** 2, axis=1)) + 1
            )
            phi = np.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.nparticles - 5 + 2 * i] = phi / (2 * np.pi) + (phi < 0)

        # get additional random numbers for the pdfs
        r1, r2 = self._get_pdf_random_numbers([x1, x2])
        r = np.concatenate((r1, r2, r), axis=-1)

        return r

    def jac(self, xs):
        return np.ones(xs.shape[0])

    def pdf(self, xs):
        nparticles = self.nparticles
        e_cm = self.e_cm
        if nparticles is None:
            nparticles = xs.shape[1] // 4

        vol = (
            (np.pi / 2.0) ** (nparticles - 1)
            * e_cm ** (2 * nparticles - 4)
            / (gamma(nparticles) * gamma(nparticles - 1))
        )
        return vol

    def pdf_gradient(self, xs):
        return 0


class Mahambo(PhaseSpaceMapping):
    """Massive hadronic Rambo on diet (Mahambo) algorithm as preprocessing

    Adopted from
        [1] Rambo [Comput. Phys. Commun. 40 (1986) 359-373]
        [2] Rambo on diet [1308.2922]

    Note, that there is an error in the algorithm of [2], which has been fixed.
    For details see
        - "How to GAN - Novel simulation methods for the LHC", PhD thesis
        (10.11588/heidok.00029154)

    There is also an error in the derivation of the massive weight in [1].
    For details see appendix of [23XX.xxxx] (our paper).

    """

    def __init__(
        self,
        e_had: float,
        n_particles: int,
        masses: list = None,
        nu: float = 0.95,
        normal_latent: bool = True,
        epsilon=1e-8,
    ):
        """
        Args:
            e_had (float): hadronic center of mass energy.
            n_particles (int): number of final state particles.
            masses (list, optional): list of final state masses. Defaults to None.
        """
        super().__init__(
            input_shape=(n_particles, 4),
            output_shape=(3 * n_particles - 2,),
            invertible=True,
            has_jacobian=True,
        )

        # self.e_had = e_had
        self.n_particles = n_particles
        self.normal_latent = normal_latent
        self.epsilon = epsilon

        # Make sure the list is as long as the number of
        # finals state particles if given
        if masses is not None:
            self.masses = torch.tensor(masses)
            assert len(self.masses) == self.n_particles
        else:
            self.masses = masses

        # Define min energy due to masses (no cuts etc)
        e_min = torch.sum(self.masses) if self.masses is not None else torch.tensor(0.0)
        # self.tau_min = (e_min / self.e_had) ** 2
        self.s_max = e_had**2
        self.s_min = e_min**2

        # Define parameter for lumi integration
        self.nu = nu

    def transform(
        self,
        x_or_p: torch.Tensor,
        rev: bool,
        jac: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if rev:
            p, logjac = self.rand_to_momenta(x_or_p)
            return p, logjac

        x, logjac = self.momenta_to_rand(x_or_p)
        return x, logjac

    def _rap_to_rand(
        self,
        s: torch.Tensor,
        r2_or_rap: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = torch.log(s / self.s_max)
        if inverse:
            rap = (r2_or_rap - 0.5) * scale
            return rap, scale.abs().log().squeeze()

        r2 = r2_or_rap / scale + 0.5
        return r2, -scale.abs().log().squeeze()

    def _energy_to_rand(
        self,
        r1_or_s: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if inverse:
            s, logdet = massless_propogator(
                r1_or_s, self.s_min, self.s_max, self.nu, inverse=True
            )
            return s, logdet.squeeze()

        r1, logdet = massless_propogator(
            r1_or_s, self.s_min, self.s_max, self.nu, inverse=False
        )
        return r1, logdet.squeeze()

    def _lumi_to_rand(
        self,
        r1_or_s: torch.Tensor,
        r2_or_rap: torch.Tensor,
        inverse: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if inverse:
            s, ldj1 = self._energy_to_rand(r1_or_s, inverse=True)
            rap, ldj2 = self._rap_to_rand(s, r2_or_rap, inverse=True)
            return s, rap, ldj1 + ldj2

        r1, ldj1 = self._energy_to_rand(r1_or_s)
        r2, ldj2 = self._rap_to_rand(r1_or_s, r2_or_rap)
        return r1, r2, ldj1 + ldj2

    def _get_rapidity_and_energy(
        self, q: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        s = q[:, :, 0] ** 2 - q[:, :, 1] ** 2 - q[:, :, 2] ** 2 - q[:, :, 3] ** 2
        rapidity = torch.arctanh(q[:, :, 3] / q[:, :, 0])
        return s, rapidity

    def rand_to_momenta(self, rs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps random numbers onto momenta

        Args:
            rs (torch.Tensor): random numbers with shape (batch, 3 * n_particles - 2)

        Returns:
            p (torch.Tensor): momenta, shape (batch, 4 * n_particles)
            logdet (torch.Tensor): log determinant of mapping, shape (batch,)
        """
        # Map onto unit-hypercube
        # Map onto gaussian random numbers
        if self.normal_latent:
            xs = torch.sigmoid(rs)
            ld_sigmoid = -torch.sum(F.softplus(-rs) + F.softplus(rs), dim=1)
        else:
            xs = rs
            ld_sigmoid = torch.zeros(xs.shape[:1], dtype=xs.dtype, device=xs.device)

        # extract parts
        xs, r1, r2 = xs[:, 2:], xs[:, [0]], xs[:, [1]]

        # get partonic energies and boost variables
        s, rapidity, ld_lumi = self._lumi_to_rand(r1, r2, inverse=True)
        e_cm = torch.sqrt(s)

        # prepare momenta
        p = torch.empty((xs.shape[0], self.n_particles, 4))
        k = torch.empty((xs.shape[0], self.n_particles, 4))

        # q = np.empty((xs.shape[0], 4))
        M = torch.zeros((xs.shape[0], self.n_particles))
        u = torch.empty((xs.shape[0], self.n_particles - 2))

        Q = e_cm * torch.tile(torch.tensor([1, 0, 0, 0]), (xs.shape[0], 1))
        Q_prev = torch.empty((xs.shape[0], 4))
        M[:, 0] = e_cm[:, 0]

        # Solve equation numerically for all directly
        func = lambda x: rambo_func(x, self.n_particles, xs)
        df = lambda x: rambo_func(x, self.n_particles, xs, diff=True)
        guess = 0.5 * torch.ones((xs.shape[0], self.n_particles - 2))
        u[:, :] = newton(func, df, 0.0, 1.0, guess)

        for i in range(2, self.n_particles + 1):
            Q_prev[:, :] = Q[:, :]
            if i != self.n_particles:
                M[:, i - 1] = torch.prod(u[:, : i - 1], dim=1) * e_cm[:, 0]

            cos_theta = 2 * xs[:, self.n_particles - 6 + 2 * i] - 1
            phi = 2 * pi * xs[:, self.n_particles - 5 + 2 * i]
            q = 4 * M[:, i - 2] * two_body_decay_factor(M[:, i - 2], M[:, i - 1], 0)

            p[:, i - 2, 0] = q
            p[:, i - 2, 1] = q * torch.cos(phi) * torch.sqrt(1 - cos_theta**2)
            p[:, i - 2, 2] = q * torch.sin(phi) * torch.sqrt(1 - cos_theta**2)
            p[:, i - 2, 3] = q * cos_theta
            Q[:, 0] = torch.sqrt(q**2 + M[:, i - 1] ** 2)
            Q[:, 1:] = -p[:, i - 2, 1:]
            p[:, i - 2] = boost(Q_prev, p[:, i - 2])
            Q = boost(Q_prev, Q)

        p[:, self.n_particles - 1] = Q

        if self.masses is not None:
            # Define masses
            m = torch.tile(self.masses, (xs.shape[0], 1))

            # solve for xi in massive case, see Ref. [1]
            xi = torch.empty((xs.shape[0], 1, 1))
            func = lambda x: mass_func(x, p, m, e_cm)
            df = lambda x: mass_func(x, p, m, e_cm, diff=True)
            guess = 0.5 * torch.ones((xs.shape[0],))
            xi[:, 0, 0] = newton(func, df, 0.0, 1.0, guess)

            # Make them massive
            k[:, :, 0] = torch.sqrt(m**2 + xi[:, :, 0] ** 2 * p[:, :, 0] ** 2)
            k[:, :, 1:] = xi * p[:, :, 1:]

            # Get jacobian and then boost into lab frame
            jac = self.weight(e_cm[:, 0], k, p, xi)
            k = boost_beam(k, rapidity, inverse=False)

            return k, jac.log() + ld_lumi + ld_sigmoid

        # Get jacobian and then boost into lab frame
        jac_rambo = self.weight(e_cm[:, 0], p, p)
        p = boost_beam(p, rapidity, inverse=False)

        return p, jac_rambo.log() + ld_lumi + ld_sigmoid

    def momenta_to_rand(self, k: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Maps momenta onto random numbers.

        Args:
            k (torch.Tensor): momenta, shape (batch, n_particles, 4)

        Returns:
            xs (torch.Tensor): random numbers, shape (batch, 3 * n_particles - 2)
            logdet (torch.Tensor): log determinant of mapping, shape (batch,)
        """
        M = torch.empty(k.shape[0])
        M_prev = torch.empty(k.shape[0])
        Q = torch.empty((k.shape[0], 4))
        r = torch.empty((k.shape[0], 3 * self.n_particles - 4))

        # Boost into partonic CM frame and get x1 and x2
        q = torch.sum(k, dim=1, keepdim=True)
        s, rapidity = self._get_rapidity_and_energy(q)
        e_cm = torch.sqrt(s)
        k = boost_beam(k, rapidity, inverse=True)

        # Make momenta massless before going back to random numbers
        p = torch.empty((k.shape[0], self.n_particles, 4))
        xi = torch.empty((k.shape[0], 1, 1))
        if self.masses is not None:
            # Define masses
            m = torch.tile(self.masses, (k.shape[0], 1))
            # solve for xi in massive case, see Ref. [1], here analytic result possible!
            xi[:, 0, 0] = (
                torch.sum(torch.sqrt(k[:, :, 0] ** 2 - m**2), dim=-1) / e_cm[:, 0]
            )
            # Make them massive
            p[:, :, 0] = torch.sqrt(k[:, :, 0] ** 2 - m**2) / xi[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:] / xi

        else:
            xi = None
            p[:, :, 0] = k[:, :, 0]
            p[:, :, 1:] = k[:, :, 1:]

        # Get jacobian
        # (its the inverse! so return with minus later for logdet!)
        jac = self.weight(e_cm[:, 0], k, p, xi)

        Q[:] = p[:, -1]

        for i in range(self.n_particles, 1, -1):
            M_prev[:] = M[:]
            P = p[:, i - 2 :].sum(dim=1)
            M = torch.sqrt(P[:, 0] ** 2 - P[:, 1] ** 2 - P[:, 2] ** 2 - P[:, 3] ** 2)

            if i != self.n_particles:
                u = M_prev / M
                r[:, i - 2] = (self.n_particles + 1 - i) * u ** (
                    2 * (self.n_particles - i)
                ) - (self.n_particles - i) * u ** (2 * (self.n_particles + 1 - i))

            Q += p[:, i - 2]
            p_prime = boost(torch.einsum("ij,ki->kj", MINKOWSKI, Q), p[:, i - 2])
            r[:, self.n_particles - 6 + 2 * i] = 0.5 * (
                p_prime[:, 3] / torch.sqrt(torch.sum(p_prime[:, 1:] ** 2, dim=1)) + 1
            )
            phi = torch.arctan2(p_prime[:, 2], p_prime[:, 1])
            r[:, self.n_particles - 5 + 2 * i] = phi / (2 * pi) + (phi < 0)

        # get additional random numbers for the pdfs
        r1, r2, ld_lumi = self._lumi_to_rand(s, rapidity)
        r = torch.concat([r1, r2, r], dim=-1)

        # Map onto gaussian random numbers
        if self.normal_latent:
            r_normal = torch.logit(r)
            ld_logit = torch.sum(F.softplus(-r_normal) + F.softplus(r_normal), dim=1)
            return r_normal, -jac.log() + ld_lumi + ld_logit

        return r, -jac.log() + ld_lumi

    def weight(
        self,
        e_cm: torch.Tensor,
        k: torch.Tensor,
        p: torch.Tensor,
        xi: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            e_cm (torch.Tensor): center-of-mass energy of partonic frame with shape (batch,)
            k (torch.Tensor): massive momenta in shape (batch, particles, 4)
            p (torch.Tensor): massless momenta in shape (batch, particles, 4)
            xi (torch.Tensor, Optional): shift variable with shape (batch,)

        Returns:
            torch.Tensor: weight of sampler
        """
        # get volume for massless particles
        w0 = (
            (pi / 2.0) ** (self.n_particles - 1)
            * e_cm ** (2 * self.n_particles - 4)
            / (gamma(self.n_particles) * gamma(self.n_particles - 1))
        )

        if xi is not None:
            xi = xi[:, 0, 0]
            # get correction factor for massive ones
            ks = torch.sqrt(k[:, :, 1] ** 2 + k[:, :, 2] ** 2 + k[:, :, 3] ** 2)
            ps = torch.sqrt(p[:, :, 1] ** 2 + p[:, :, 2] ** 2 + p[:, :, 3] ** 2)
            k0 = k[:, :, 0]
            p0 = p[:, :, 0]
            w_M = (
                xi ** (3 * self.n_particles - 3)
                * torch.prod(p0 / k0, dim=1)
                * torch.sum(ps**2 / p0, dim=1)
                / torch.sum(ks**2 / k0, dim=1)
            )
            return w0 * w_M

        return w0
