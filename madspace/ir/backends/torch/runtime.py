from typing import Callable

import torch

from . import kernels
from ...ir import IRVariable, IRDType, IRFunction


TORCH_DTYPES = {
    IRDType.float: torch.float64, IRDType.int: torch.int64, IRDType.bool: torch.bool
}


def _check_types(variables: list[IRVariable], args: list[torch.Tensor]):
    if len(variables) != len(args):
        raise ValueError(
            f"Wrong number of arguments: expected {len(variables)}, got {len(args)}"
        )
    for i, (var, arg) in zip(variables, args):
        if var.type.shape != arg.shape:
            raise ValueError(
                f"Wrong shape for argument {i}: " +
                f"expected {var.type.shape}, got {arg.shape}"
            )
        var_dtype = TORCH_DTYPES[var.type.dtype]
        if var_dtype != arg.dtype:
            raise ValueError(
                f"Wrong dtype for argument {i}: " +
                f"expected {var_dtype}, got {arg.dtype}"
            )


def compile(function: IRFunction) -> Callable:
    def torch_func(*args: torch.Tensor) -> list[torch.Tensor]:
        _check_types(function.inputs, args)
        variables = {var: arg for var, arg in zip(function.inputs, args)}
        for instruction in function.instructions:
            inputs = [variables[var] for var in instruction.inputs]
            outputs = getattr(kernels, instruction.name)(*inputs, *instruction.constants)
            variables.update(zip(instruction.outputs, outputs))
        outputs = [variables[var] for var in function.outputs]
        _check_types(function.outputs, outputs)
        return outputs
    return torch_func
