from typing import Tuple, Optional, List

import torch
import torch.nn as nn

class PhaseSpaceGenerator(nn.Module):
    """Base class for all phase-space generator objects.
    """

    def __init__(self, dims_in: int, dims_c: Optional[int]):
        super().__init__()
        self.dims_in = dims_in
        self.dims_c = dims_c

    def _check_inputs(self, x: torch.Tensor, condition: Optional[torch.Tensor]):
        if len(x.shape) != 2 or x.shape[1] != self.dims_in:
            raise ValueError(
                f"Expected input shape (?, {self.dims_in}), got {x.shape}"
            )
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
        x: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of the mapping ``f``.
        Conventionally, this is the pass from the
        momenta/data ``x`` to the latent space ``z``, i.e.
        ..math::
            f(x) = z.
        Args:
            x: Tensor with shape (batch_size, n_features).
            condition: None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
        Returns:
            z: Tensor with shape (batch_size, n_features).
            logdet: Tensor of shape (batch_size,), the logdet of the mapping.
        """
        self._check_inputs(x, condition)
        return self._forward(x, condition, **kwargs)

    def _forward(self, x, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _forward(...) method"
        )

    def inverse(
        self,
        z: torch.Tensor,
        condition: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse pass ``f^{-1}`` of the mapping. Conventionally, this is the pass
        from the random numbers ``z` to the momenta/data ``x``, i.e.
        ..math::
            f^{-1}(z) = x.
        Args:
            z: Tensor with shape (batch_size, n_features).
            condition: None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
        Returns:
            x: Tensor with shape (batch_size, n_features).
            logdet: Tensor of shape (batch_size,), the logdet of the mapping.
        """
        self._check_inputs(z, condition)
        return self._inverse(z, condition, **kwargs)

    def _inverse(self, z, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _inverse(...) method"
        )

    def det(
        self,
        x_or_z: torch.Tensor,
        condition: torch.Tensor = None,
        inverse: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Calculates the jacobian determinant of the mapping:
        ...math::
            det = |J|, with
            J = dz/dx = df(x)/dx
        or for the inverse mapping:
        ...math::
            det = |J_inv|, with
            J_inv = dx/dz = df^{-1}(z)/dz
        Args:
            x_or_z: Tensor with shape (batch_size, n_features).
            condition (optional): None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): return det of inverse pass. Defaults to False.
        Returns:
            det: Tensor of shape (batch_size,).
        """
        self._check_inputs(x_or_z, condition)
        return self._det(x_or_z, condition, inverse, **kwargs)
    
    def _det(self, x, condition, inverse, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _inverse(...) method"
        )
