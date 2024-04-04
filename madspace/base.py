from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn


class PhaseSpaceMapping(nn.Module):
    """Base class for all mapping objects."""

    def __init__(
        self,
        dims_r: Tuple[int],
        dims_p: Tuple[int],
        dims_c: Optional[Tuple[int]] = None,
    ):
        super().__init__()
        self.dims_r = dims_r
        self.dims_p = dims_p
        self.dims_c = dims_c

    def _check_inputs(
        self,
        r_or_p: Tensor,
        condition: Optional[Tensor] = None,
        inverse: bool = False,
    ) -> None:
        dims_in = self.dims_r if inverse else self.dims_p
        if r_or_p.shape[1:] != self.dims_r:
            raise ValueError(
                f"Expected input shape {dims_in}, but got {tuple(r_or_p.shape[1:])}"
            )
        if self.dims_c is None:
            return
        if condition is None:
            raise ValueError("Expected condition")
        else:
            if condition.shape[1:] != self.dims_c:
                raise ValueError(
                    f"Expected condition shape {self.dims_c}, got {tuple(condition.shape[1:])}"
                )
            if r_or_p.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )

    def forward(
        self,
        p: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the mapping ``f``.
        Conventionally, this is the pass from the
        momenta ``p`` to the random numbers ``r``, i.e.
        ..math::
            f(p) = r.
        Args:
            p: Tensor with shape=(b, np, 4).
            condition: None or Tensor with shape=(b, dc1, [dc2,...]).
                If None, the condition is ignored. Defaults to None.
        Returns:
            r: Tensor with shape=(b, nr).
            logdet: Tensor of shape=(b,), the logdet of the mapping.
        """
        self._check_inputs(p, condition)
        return self._forward(p, condition, **kwargs)

    def _forward(self, p, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _forward(...) method"
        )

    def inverse(
        self,
        r: Tensor,
        condition: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Inverse pass ``f^{-1}`` of the mapping. Conventionally, this is the pass
        from the random numbers ``r` to the momenta ``p``, i.e.
        ..math::
            f^{-1}(r) = p.
        Args:
            r: Tensor with shape=(b, nr).
            condition: None or Tensor with shape=(b, dc1, [dc2,...]).
                If None, the condition is ignored. Defaults to None.
        Returns:
            p: Tensor with shape=(b, np, 4).
            logdet: Tensor of shape=(b,), the logdet of the mapping.
        """
        self._check_inputs(r, condition, inverse=True)
        return self._inverse(r, condition, **kwargs)

    def _inverse(self, r, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _inverse(...) method"
        )

    def log_det(
        self,
        p_or_r: Tensor,
        condition: Tensor = None,
        inverse: bool = False,
        **kwargs,
    ) -> Tensor:
        """Calculate log det of the mapping only:
        ...math::
            log_det = log(|J|), with
            J = dr/dp = df(p)/dp
        or for the inverse mapping:
        ...math::
            log_det = log(|J_inv|), with
            J_inv = dp/dr = df^{-1}(r)/dr
        Args:
            p_or_z: Tensor with shape=(b, np, 4) or shape=(b, nr).
            condition (optional): None or Tensor with shape=(b, dc1, [dc2,...]).
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.
        Returns:
            log_det: Tensor of shape (b,).
        """
        self._check_inputs(p_or_r, condition, inverse=inverse)
        return self._call_log_det(p_or_r, condition, inverse, **kwargs)

    def _call_log_det(self, x, condition, inverse, **kwargs):
        """Wrapper around _log_det."""
        if hasattr(self, "_log_det"):
            return self._log_det(x, condition, inverse, **kwargs)
        if hasattr(self, "_det"):
            return torch.log(self._det(x, condition, inverse, **kwargs))
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide log_det(...) method"
        )

    def det(
        self,
        p_or_r: Tensor,
        condition: Tensor = None,
        inverse: bool = False,
        **kwargs,
    ) -> Tensor:
        """Calculates the jacobian determinant of the mapping:
        ...math::
            det = |J|, with
            J = dr/dp = df(p)/dp
        or for the inverse mapping:
        ...math::
            det = |J_inv|, with
            J_inv = dp/dr = df^{-1}(r)/dr
        Args:
            p_or_z: Tensor with shape=(b, np, 4) or shape=(b, nr).
            condition (optional): None or Tensor with shape=(b, dc1, [dc2,...]).
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.
        Returns:
            det: Tensor of shape (b,).
        """
        self._check_inputs(p_or_r, condition, inverse=inverse)
        return self._call_det(p_or_r, condition, inverse, **kwargs)

    def _call_det(self, x, condition, inverse, **kwargs):
        """Wrapper around _det."""
        if hasattr(self, "_det"):
            return self._det(x, condition, inverse, **kwargs)
        if hasattr(self, "_log_det"):
            return torch.exp(self._log_det(x, condition, inverse, **kwargs))
        raise NotImplementedError("det is not implemented")
