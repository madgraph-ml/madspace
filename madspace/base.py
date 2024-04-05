from typing import Tuple, Optional

import torch
from torch import Tensor
import torch.nn as nn

# Definition of InputTypes and OutputTypes
ShapeList = list[Tuple[int]]
TensorList = list[Tensor]
TensorTuple = Tuple[Tensor]


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
        dims_in: ShapeList,
        dims_out: ShapeList,
        dims_c: Optional[ShapeList] = None,
        debug: bool = False,
    ):
        """
        Args:
            dims_in (ShapeList): list of input shapes for the forward map w/o batch dimension ``b``.
                Includes random numbers r and potential auxiliary inputs.
            dims_out (ShapeList): list of output shapes as inputs for inverse map w/o batch dimension ``b``.
                Usually only includes a sigle shape from the momentum tensor.
            dims_c (ShapeList, optional): list of shapes for the conditions. Defaults to None.
            debug (bool, optional): check shapes of all inputs. Defaults to False.
        """
        super().__init__()
        self.dims_in = dims_in
        self.dims_out = dims_out
        self.dims_c = dims_c
        self.debug = debug

    def _check_inputs(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        inverse: bool = False,
    ) -> None:
        """Checks if inputs have the correct formats

        Args:
            inputs (TensorList): inputs for foward map or inverse map shapes dims_in and dims_out
            condition (TensorList, optional): conditional inputs with shapelist [shape_c1, shape_c2,...].
                If None, the condition is ignored. Defaults to None.
            inverse (bool, optional): check inverse map. Defaults to False.

        Raises:
            ValueError: raises error when inputs do not have correct dimensions
        """
        if not inverse:
            dims_in_range = range(len(self.dims_in))
            in_dim_list = list(tuple(inputs[i].shape[1:]) for i in dims_in_range)
            if any(inputs[i].shape[1:] != self.dims_c[i] for i in dims_in_range):
                raise ValueError(
                    f"Expected input shape {self.dims_in}, but got {in_dim_list}"
                )
        else:
            dims_out_range = range(len(self.dims_out))
            out_dim_list = list(tuple(inputs[i].shape[1:]) for i in dims_out_range)
            if any(inputs[i].shape[1:] != self.dims_c[i] for i in dims_out_range):
                raise ValueError(
                    f"Expected input shape {self.dims_out}, but got {out_dim_list}"
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
            if any(inputs[i].shape[0] != condition[i].shape[0] for i in dimc_range):
                raise ValueError(
                    "Number of input items must be equal to number of any condition item."
                )

    def forward(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[TensorTuple, Tensor]:
        """
        Forward pass of the ps-mapping ``f``. This is the pass
        from the random numbers ``r` to the momenta ``p``, i.e.
            ..math::
                f(r) = p.
        Args:
            inputs (TensorList): forward map inputs with shapes=[(b, *dims_in0), (b, *dims_in1),...].
            condition (TensorList, optional): conditional inputs. Defaults to None.

        Returns:
            out (TensorTuple): tuple including momenta with shape=(b, *dims_out0).
            logdet (Tensor): the logdet of the mapping shape=(b,).
        """
        if self.debug:
            self._check_inputs(inputs, condition)
        return self._map(inputs, condition, **kwargs)

    def _map(self, inputs, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map(...) method"
        )

    def inverse(
        self,
        inputs: TensorList,
        condition: Optional[TensorList] = None,
        **kwargs,
    ) -> Tuple[TensorTuple, Tensor]:

        """
        Inverse pass ``f^{-1}`` of the ps-mapping. This is the pass
        from the momenta ``p`` to the random numbers ``r``, i.e.
        ..math::
            f^{-1}(p) = r.
        Args:
            inputs (TensorList): input list including momenta with shape=(b, *dims_out0).
            condition (TensorList, optional): conditional inputs. Defaults to None.
        Returns:
            out (TensorTuple): tuple including randon numbers with shape=(b, *dims_in0)
                and additional auxiliary tensors with shapelist=[(b, *dims_in1),..]
            logdet (Tensor): logdet of the mapping with shape=(b,).
        """
        if self.debug:
            self._check_inputs(inputs, condition, inverse=True)
        return self.map_inverse(inputs, condition, **kwargs)

    def _map_inverse(self, p, condition, **kwargs):
        """Should be overridden by all subclasses."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not provide _map_inverse(...) method"
        )

    def log_det(
        self,
        inputs: TensorList,
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
        if self.debug:
            self._check_inputs(inputs, condition, inverse=inverse)
        return self._call_log_det(inputs, condition, inverse, **kwargs)

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
        inputs: TensorList,
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
        if self.debug:
            self._check_inputs(inputs, condition, inverse=inverse)
        return self._call_det(inputs, condition, inverse, **kwargs)

    def _call_det(self, x, condition, inverse, **kwargs):
        """Wrapper around _det."""
        if hasattr(self, "_det"):
            return self._det(x, condition, inverse, **kwargs)
        if hasattr(self, "_log_det"):
            return torch.exp(self._log_det(x, condition, inverse, **kwargs))
        raise NotImplementedError("det is not implemented")
