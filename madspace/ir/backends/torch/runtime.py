from typing import Callable
from collections import defaultdict

import torch

from . import kernels
from ...ir import IRVariable, IRDType, IRFunction


TORCH_DTYPES = {
    IRDType.float: torch.float64, IRDType.int: torch.int64, IRDType.bool: torch.bool
}


_debug_mode = True
def set_debug_mode(debug_mode: bool):
    _debug_mode = debug_mode


def _check_types(
    variables: list[IRVariable], args: list[torch.Tensor], is_arg: bool, prefix: str = ""
):
    obj_str = "argument" if is_arg else "output"
    if len(variables) != len(args):
        raise ValueError(
            f"{prefix}wrong number of {obj_str}s: expected {len(variables)}, got {len(args)}"
        )
    for i, (var, arg) in enumerate(zip(variables, args)):
        if var.type.shape != arg.shape[1:]:
            raise ValueError(
                f"{prefix}wrong shape for {obj_str} {i}: " +
                f"expected {var.type.shape}, got {arg.shape[1:]}"
            )
        var_dtype = TORCH_DTYPES[var.type.dtype]
        if var_dtype != arg.dtype:
            raise ValueError(
                f"{prefix}wrong dtype for {obj_str} {i}: " +
                f"expected {var_dtype}, got {arg.dtype}"
            )


def _compile_debug(function: IRFunction) -> Callable:
    def torch_func(*args: torch.Tensor) -> list[torch.Tensor]:
        _check_types(function.inputs, args, True)
        variables = {var: arg for var, arg in zip(function.inputs, args)}
        for i, instruction in enumerate(function.instructions):
            inputs = [variables[var] for var in instruction.inputs]
            prefix = f"line {i+1}, {instruction.instruction.name}: "
            _check_types(instruction.inputs, inputs, True, prefix)
            outputs = getattr(kernels, instruction.instruction.name)(
                *inputs, *instruction.constants
            )
            if not isinstance(outputs, tuple):
                outputs = (outputs, )
            _check_types(instruction.outputs, outputs, False, prefix)
            variables.update(zip(instruction.outputs, outputs))
            #print(instruction, inputs, outputs)
        outputs = [variables[var] for var in function.outputs]
        _check_types(function.outputs, outputs, False)
        return outputs
    return torch_func


def _compile_fast(function: IRFunction) -> Callable:
    tmp_count = 0
    def tmp_factory():
        nonlocal tmp_count
        value = f"tmp{tmp_count}"
        tmp_count += 1
        return value
    var_names = defaultdict(tmp_factory)
    var_names.update({in_var: f"input{i}" for i, in_var in enumerate(function.inputs)})
    var_names.update({out_var: f"output{i}" for i, out_var in enumerate(function.outputs)})
    param_str = ", ".join(f"input{i}" for i in range(len(function.inputs)))
    lines = [f"def torch_func({param_str}):"]
    for i, instr in enumerate(function.instructions):
        inputs = ", ".join(
            [var_names[var] for var in instr.inputs] +
            [repr(const) for const in instr.constants]
        )
        output_names = ", ".join(var_names[var] for var in instr.outputs)
        lines.append(f"    {output_names} = {instr.instruction.name}({inputs})")
    return_str = ", ".join(f"output{i}" for i in range(len(function.outputs)))
    lines.append(f"    return {return_str}")

    glob = {**vars(kernels), "_check_types": _check_types}
    exec("\n".join(lines), glob)
    return glob["torch_func"]


def compile(function: IRFunction) -> Callable:
    if _debug_mode:
        return _compile_debug(function)
    else:
        return _compile_fast(function)
