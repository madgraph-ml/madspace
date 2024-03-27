from typing import Tuple, Optional, Union

import torch
import torch.nn as nn


class PhaseSpaceMapping(nn.Module):
    """Base class for all phase-space generator objects."""

    def __init__(
        self,
        dims_in: int,
        dims_out: Optional[int] = None,
        dims_c: Optional[int] = None,
        invertible: bool = False,
    ):
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out if dims_out is not None else dims_in
        self.dims_c = dims_c
        self.invertible = invertible

    def _check_inputs(
        self,
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        rev: bool = False,
    ):
        check_dims = self.dims_in if not rev else self.dims_out
        if len(x.shape) != 2 or x.shape[1] != check_dims:
            raise ValueError(f"Expected input shape (?, {check_dims}), got {x.shape}")
        if self.dims_c is None:
            return
        if condition is None:
            raise ValueError("Expected condition")
        else:
            if len(condition.shape) != 2 or condition.shape[1] != self.dims_c:
                raise ValueError(
                    f"Expected condition shape (?, {self.dims_c}), got {condition.shape}"
                )
            if x.shape[0] != condition.shape[0]:
                raise ValueError(
                    "Number of input items must be equal to number of condition items."
                )

    def forward(
        self,
        p_or_z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        rev: bool = False,
        jac: bool = False,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        if rev and not self.invertible:
            raise ValueError("Tried to call inverse of non-invertible transformation")
        self._check_inputs(p_or_z, rev)
        if rev:
            return self.inv_map(p_or_z, condition=condition, jac=jac, **kwargs)
        return self.map(p_or_z, condition=condition, jac=jac, **kwargs)

    def map(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        jac: bool,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Should be overridden by all subclasses."""
        raise NotImplementedError

    def inv_map(
        self,
        z: torch.Tensor,
        condition: torch.Tensor,
        jac: bool,
        **kwargs,
    ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Should be overridden by all subclasses."""
        raise NotImplementedError

    def pdf(self, z: torch.Tensor) -> torch.Tensor:
        """Should be overridden by all subclasses."""
        raise NotImplementedError

    def pdf_gradient(self, z: torch.Tensor) -> torch.Tensor:
        """Should be overridden by all subclasses."""
        raise NotImplementedError
