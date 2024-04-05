from typing import Tuple, Optional, List

import torch
from torch import Tensor
import torch.nn as nn

# Definition of InputTypes
Shape = Tuple[int]
ShapeList = List[Tuple[int]]
TensorList = List[Tensor]


class PhaseSpaceMapping(nn.Module):
    """Base class for all phase-space mappings.

    Note:
    This is not a 1:1 replacement of the Mapping class in MadNIS
    as it asks for more shape information and also allows for
    multiple conditions. Further, to make it more easy as a reader.
    the forward pass denotes the mapping from the random numbers to
    the moment and vice versa, i.e.
        ..math::
            forward: f(r) = p.
            inverse: f^{-1}(p) = r.
    """

    def __init__(
        self,
        dims_r: Shape,
        dims_p: Shape,
        dims_c: Optional[ShapeList] = None,
    ):
        """
        Args:
            dims_r (Shape): shape of the random numbers w/o batch dimension ``b``.
            dims_p (Shape): shape of the input momenta w/o batch dimension ``b``.
            dims_c (ShapeList, optional): list of shapes for the conditions. Defaults to None.
        """
        super().__init__()
        self.dims_r = dims_r
        self.dims_p = dims_p
        self.dims_c = dims_c

    def _check_inputs(
        self,
        r_or_p: Tensor,
        condition: Optional[TensorList] = None,
        inverse: bool = False,
    ) -> None:
        """Checks if inputs have the correct formats

        Args:
            r_or_p (Tensor): input random numbers or momenta with shapes=(b, *dim_r/p)
            condition (TensorList, optional): conditional inputs with shapelist [shape_c1, shape_c2,...].
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): check inverse map. Defaults to False.

        Raises:
            ValueError: raises error when inputs do not have correct dimensions
        """
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
            dimc_range = range(len(self.dims_c))
            cdim_list = list(tuple(condition[i].shape[1:]) for i in dimc_range)
            if any(condition[i].shape[1:] != self.dims_c[i] for i in dimc_range):
                raise ValueError(
                    f"Expected condition shape {self.dims_c}, got {cdim_list}"
                )
            if any(r_or_p.shape[0] != condition[i].shape[0] for i in dimc_range):
                raise ValueError(
                    "Number of input items must be equal to number of any condition item."
                )

    def forward(
        self,
        r: Tensor,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:
        """
        Forward pass of the ps-mapping ``f``. This is the pass
        from the random numbers ``r` to the momenta ``p``, i.e.
            ..math::
                f(r) = p.

        Args:
            r (Tensor): random number input with shape=(b, *dim_r).
            condition (TensorList, optional): conditional inputs with shapelist [shape_c1, shape_c2,...].
                If None, the condition is ignored. Defaults to None.

        Returns:
            p (Tensor): output momenta with shape=(b, *dims_p).
            logdet (Tensor): the logdet of the mapping shape=(b,).
        """
        self._check_inputs(r, condition)
        return self._map(r, condition, **kwargs)

    def _map(self, r, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map(...) method"
        )

    def inverse(
        self,
        p: Tensor,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[Tensor, Tensor]:

        """
        Inverse pass ``f^{-1}`` of the ps-mapping. This is the pass
        from the momenta ``p`` to the random numbers ``r``, i.e.
        ..math::
            f^{-1}(p) = r.
        Args:
            p (Tensor) : input momenta with shape=(b, *dim_p).
            condition (TensorList, optional): conditional inputs with shapelist [shape_c1, shape_c2,...].
                If None, the condition is ignored. Defaults to None.
        Returns:
            r (Tensor): randon numbers with shape=(b, *dim_r).
            logdet (Tensor): logdet of the mapping with shape=(b,).
        """
        self._check_inputs(p, condition, inverse=True)
        return self.map_inverse(p, condition, **kwargs)

    def _map_inverse(self, p, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map_inverse(...) method"
        )

    def log_det(
        self,
        p_or_r: Tensor,
        condition: Optional[TensorList] = None,
        inverse: bool = False,
        **kwargs,
    ) -> Tensor:
        """
        Calculate log det of the mapping only:
            ..math::
                log_det = log(|J|), with
                J = dp/dr = df(r)/dr
        or for the inverse mapping:
            ..math::
                log_det = log(|J_inv|), with
                J_inv = dr/dp = df^{-1}(p)/dr
        Args:
            ...see forward/inverse for first 2 inputs
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.
        Returns:
            log_det: Tensor of shape (b,).
        """
        self._check_inputs(p_or_r, condition, inverse=inverse)
        return self._call_log_det(p_or_r, condition, inverse, **kwargs)

    def _call_log_det(self, x, condition, inverse, **kwargs) -> Tensor:
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
        condition: Optional[TensorList] = None,
        inverse: bool = False,
        **kwargs,
    ) -> Tensor:
        """Calculates the jacobian determinant of the mapping:
        ...math::
            det = |J|, with
            J = dp/dr = df(r)/dr
        or for the inverse mapping:
        ...math::
            det = |J_inv|, with
            J_inv = dr/dp = df^{-1}(p)/dp
        Args:
            ...see forward/inverse for first 2 inputs
            inverse (bool, optional): return logdet of inverse pass. Defaults to False.
        Returns:
            det: Tensor of shape (b,).
        """
        self._check_inputs(p_or_r, condition, inverse=inverse)
        return self._call_det(p_or_r, condition, inverse, **kwargs)

    def _call_det(self, p_or_r, condition, inverse, **kwargs):
        """Wrapper around _det."""
        if hasattr(self, "_det"):
            return self._det(p_or_r, condition, inverse, **kwargs)
        if hasattr(self, "_log_det"):
            return torch.exp(self._log_det(p_or_r, condition, inverse, **kwargs))
        raise NotImplementedError("det is not implemented")
