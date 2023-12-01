""" Helper functions needed for phase-space mappings """

import torch

def kaellen(a: torch.Tensor, b: torch.Tensor, c: torch.Tensor):
    return a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a


def boost(q, ph):
    metric = torch.diag([1.0, -1.0, -1.0, -1.0])
    rsq = torch.sqrt(torch.einsum("kd,dd,kd->k", q, metric, q))

    p0 = torch.einsum("ki,ki->k", q, ph) / rsq
    c1 = (ph[:, 0] + p0) / (rsq + q[:, 0])
    px = ph[:, 1] + c1 * q[:, 1]
    py = ph[:, 2] + c1 * q[:, 2]
    pz = ph[:, 3] + c1 * q[:, 3]
    p = torch.stack((p0, px, py, pz), dim=-1)

    return p


def boost_z(q, rapidity, inverse=False):
    sign = -1.0 if inverse else 1.0

    pi0 = q[:, :, 0] * torch.cosh(rapidity) + sign * q[:, :, 3] * torch.sinh(
        rapidity
    )
    pix = q[:, :, 1]
    piy = q[:, :, 2]
    piz = q[:, :, 3] * torch.cosh(rapidity) + sign * q[:, :, 0] * torch.sinh(
        rapidity
    )
    p = torch.stack((pi0, pix, piy, piz), axis=-1)

    return p
