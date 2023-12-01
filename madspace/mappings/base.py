from typing import Tuple, Optional, List

import torch
import torch.nn as nn

class Mapping(nn.Module):
    """Base class for all mapping objects.
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

    def log_det(
        self,
        x_or_z: torch.Tensor,
        condition: torch.Tensor = None,
        inverse: bool = False,
        **kwargs
    ) -> torch.Tensor:
        """Calculate log det of the mapping only:
        ...math::
            log_det = log(|J|), with
            J = dz/dx = df(x)/dx
        or for the inverse mapping:
        ...math::
            log_det = log(|J_inv|), with
            J_inv = dx/dz = df^{-1}(z)/dz
        Args:
            x_or_z: Tensor with shape (batch_size, n_features).
            condition (optional): None or Tensor with shape (batch_size, n_features).
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.
        Returns:
            log_det: Tensor of shape (batch_size,).
        """
        self._check_inputs(x_or_z, condition)
        return self._call_log_det(x_or_z, condition, inverse, **kwargs)

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
        return self._call_det(x_or_z, condition, inverse, **kwargs)

    def _call_det(self, x, condition, inverse, **kwargs):
        """Wrapper around _det."""
        if hasattr(self, "_det"):
            return self._det(x, condition, inverse, **kwargs)
        if hasattr(self, "_log_det"):
            return torch.exp(self._log_det(x, condition, inverse, **kwargs))
        raise NotImplementedError("det is not implemented")

    def apply_mapping(self, mapping: 'Mapping') -> 'ChainedMapping':
        dims_c = mapping.dims_c if self.dims_c is None else self.dims_c
        return ChainedMapping(self.dims_in, self.dims_c, [mapping, self])


class ConditionalMapping(Mapping):
    def __init__(
        self,
        dims_in: int,
        dims_c: Optional[int],
        condition_mask: Optional[torch.Tensor]
    ):
        super().__init__(dims_in, dims_c)
        if condition_mask is None:
            condition_mask = torch.ones(dims_c, dtype=torch.bool)
        elif condition_mask.shape != (dims_c,):
            raise ValueError(f"Condition mask must have shape (dims_c,)")
        elif condition_mask.dtype != torch.bool:
            raise ValueError(f"Condition mask must be boolean tensor")
        self.register_buffer("condition_mask", condition_mask)


class InverseMapping(Mapping):
    def __init__(self, dims_in: int, dims_c: Optional[int], mapping: Mapping):
        super().__init__(dims_in, dims_c)
        if mapping.dims_in != dims_in:
            raise ValueError("Mapping input dimensions incompatible")
        if mapping.dims_c != dims_c:
            raise ValueError("Mapping condition dimensions incompatible")
        self.mapping = mapping

    def _forward(self, x, condition, **kwargs):
        return self.mapping.inverse(x, condition, **kwargs)

    def _inverse(self, z, condition, **kwargs):
        return self.mapping.forward(z, condition, **kwargs)

    def _log_det(self, x_or_z, condition=None, inverse=False, **kwargs):
        return self.mapping.log_det(x_or_z, condition, not inverse, **kwargs)


class ChainedMapping(Mapping):
    def __init__(self, dims_in: int, dims_c: Optional[int], mappings: List[Mapping]):
        super().__init__(dims_in, dims_c)
        flattened_mappings = []
        for m in mappings:
            if dims_in != m.dims_in:
                raise ValueError("Incompatible input dimension")
            if m.dims_c is not None and dims_c != m.dims_c:
                raise ValueError("Incompatible condition dimension")
            if isinstance(m, ChainedMapping):
                flattened_mappings.extend(m.mappings)
            else:
                flattened_mappings.append(m)
        self.mappings = nn.ModuleList(flattened_mappings)

    def _forward(self, x, condition, **kwargs):
        log_jac_all = 0.
        for mapping in self.mappings:
            x, log_jac = mapping.forward(x, condition, **kwargs)
            log_jac_all += log_jac
        return x, log_jac_all

    def _inverse(self, z, condition, **kwargs):
        log_jac_all = 0.
        for mapping in reversed(self.mappings):
            z, log_jac = mapping.inverse(z, condition, **kwargs)
            log_jac_all += log_jac
        return z, log_jac_all

    def _log_det(self, x_or_z, condition=None, inverse=False, **kwargs):
        return sum(mapping.log_det(x_or_z, condition, inverse, **kwargs) for mapping in self.mappings)
