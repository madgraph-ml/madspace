from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import defaultdict
import importlib


class IRDType(Enum):
    float = 0
    int = 1
    bool = 2

    def __repr__(self):
        return self.name

    def __str__(self):
        return self.name


@dataclass
class IRType:
    shape: tuple[int, ...]
    dtype: IRDType = IRDType.float


scalar = IRType(shape=())
scalar_int = IRType(shape=(), dtype=IRDType.int)
four_vector = IRType(shape=(4,))
four_vector_array = lambda size: IRType(shape=(size, 4))


@dataclass(eq=False)
class IRVariable:
    type: IRType

    def __hash__(self):
        return id(self)


Constant = float | int | bool | list[float] | list[int] | list[bool]
ArgType = IRVariable | Constant


@dataclass
class IRInstruction:
    name: str
    input_dtypes: list[IRDType, ...]
    input_shapes: list[tuple[int | str, ...], ...]
    constant_types: list[Type]
    output_dtypes: list[IRDType, ...]
    output_shapes: list[tuple[int | str, ...], ...]

    def eval_types(self, args: list[ArgType]) -> list[IRType]:
        assert len(self.input_dtypes) == len(self.input_shapes)
        assert len(self.output_dtypes) == len(self.output_shapes)

        #TODO: check length
        #TODO: check constant types
        variables = {}
        for i, (input_dtype, input_shape, arg) in enumerate(
            zip(self.input_dtypes, self.input_shapes, args)
        ):
            # TODO: also check dtype
            if input_dtype != arg.type.dtype:
                raise ValueError(f"Input {i}: Expected {input_dtype}, got {arg.type.dtype}")
            if len(input_shape) != len(arg.type.shape):
                raise ValueError(
                    f"Input {i}: Expected dimension {len(input_shape)}, " +
                    f"got {len(arg.type.shape)}"
                )
            for input_shape_item, arg_shape_item in zip(input_shape, arg.type.shape):
                if isinstance(input_shape_item, str):
                    if input_shape_item.isidentifier() and input_shape_item not in variables:
                        variables[input_shape_item] = arg_shape_item
                    elif eval(input_shape_item, variables) != arg_shape_item:
                        raise ValueError(f"Input {i}: shape mismatch")
                elif input_shape_item != arg_shape_item:
                    raise ValueError(f"Input {i}: shape mismatch")

        return [
            IRType(
                dtype = output_dtype,
                shape = tuple(
                    eval(output_shape_item, variables)
                    if isinstance(output_shape_item, str)
                    else output_shape_item
                    for output_shape_item in output_shape
                ),
            ) for output_dtype, output_shape in zip(self.output_dtypes, self.output_shapes)
        ]

    def __call__(self, *args: IRVariable) -> IRInstructionCall:
        output_types = self.eval_types(args)
        n_inputs = len(args) - len(self.constant_types)
        return IRInstructionCall(
            instruction = self,
            inputs = list(args[:n_inputs]),
            constants = list(args[n_inputs:]),
            outputs = list(map(IRVariable, output_types)),
        )


@dataclass
class IRInstructionCall:
    instruction: IRInstruction
    inputs: list[IRVariable]
    constants: list[Constant]
    outputs: list[IRVariable]

    def __str__(self):
        in_str = ",".join(str(id(var)) for var in self.inputs)
        out_str = ",".join(str(id(var)) for var in self.outputs)
        return f"{out_str} = {self.instruction.name}({in_str})"

class IRFunction:
    def __init__(self, input_types: list[IRType], output_types: list[IRType]):
        self.instructions: list[IRInstructionCall] = []
        self.inputs: list[IRVariable] = [IRVariable(ir_type) for ir_type in input_types]
        self.output_types: list[IRType] = output_types
        self.outputs: list[IRVariable] = [None] * len(output_types)
        self._compiled_function = None

    def compile(self, backend=None):
        self._check_outputs()
        if backend is None:
            backend = default_backend
        return backend.compile(self)

    def __call__(self, *args, backend=None):
        if self._compiled_function is None:
            self._compiled_function = self.compile(backend)
        return self._compiled_function(*args)

    def __getattr__(self, name: str):
        if name not in INSTRUCTION_SET:
            raise AttributeError(f"Unknown instruction '{name}'")

        instruction = INSTRUCTION_SET[name]
        def add_instr(*args: ArgType):
            instr = instruction(*args)
            self.instructions.append(instr)
            return instr.outputs if len(instr.outputs) != 1 else instr.outputs[0]
        return add_instr

    def __str__(self):
        tmp_count = 0
        def tmp_factory():
            nonlocal tmp_count
            value = f"tmp{tmp_count}"
            tmp_count += 1
            return value
        var_names = defaultdict(tmp_factory)
        var_names.update({in_var: f"input{i}" for i, in_var in enumerate(self.inputs)})
        var_names.update({out_var: f"output{i}" for i, out_var in enumerate(self.outputs)})
        lines = []
        for instr in self.instructions:
            inputs = ", ".join(
                [var_names[var] for var in instr.inputs] +
                [str(const) for const in instr.constants]
            )
            output_names = ", ".join(var_names[var] for var in instr.outputs)
            lines.append(f"{output_names} = {instr.instruction.name}({inputs})")
        return "\n".join(lines)

    def _check_outputs(self):
        for i, (output, output_type) in enumerate(zip(self.outputs, self.output_types)):
            if output is None:
                raise ValueError(f"No value for output {i}")
            if output.type != output_type:
                raise ValueError(f"Output {i}: Expected type {output_type}, got {output.type}")

    def replace_var(self, old_var: IRVariable, new_var: IRVariable):
        assert old_var.type == new_var.type
        def replace_in_list(var_list):
            for i in range(len(var_list)):
                if var_list[i] == old_var:
                    var_list[i] = new_var

        replace_in_list(self.inputs)
        replace_in_list(self.outputs)
        for instr in self.instructions:
            replace_in_list(instr.inputs)
            replace_in_list(instr.outputs)

    def optimize(self):
        pass

    @staticmethod
    def join_channels(functions: list[IRFunction]) -> IRFunction:
        input_types = [var.type for var in functions[0].inputs]
        output_types = functions[0].output_types

        for func in functions:
            if len(func.inputs) != len(input_types):
                raise ValueError("All channels must have the same number of inputs")
            if len(func.outputs) != len(output_types):
                raise ValueError("All channels must have the same number of outputs")
            if any(var.type != in_type for var, in_type in zip(func.inputs, input_types)):
                raise ValueError("All channels must have the same input types")
            if any(var.type != out_type for var, out_type in zip(func.outputs, output_types)):
                raise ValueError("All channels must have the same output types")

        out_func = IRFunction(
            input_types=[*input_types, scalar_int], output_types=output_types
        )
        channel_inputs = [
            out_func.batch_split(in_var, out_func.inputs[-1], len(functions))
            for in_var in out_func.inputs[:-1]
        ]
        for i, func in enumerate(functions):
            out_func.instructions.extend(func.instructions)
            for func_input, channel_input in zip(func.inputs, channel_inputs):
                out_func.replace_var(func_input, channel_input[i])
        for i in range(len(output_types)):
            out_func.outputs[i] = out_func.batch_cat(
                *(func.outputs[i] for func in functions), out_func.inputs[-1]
            )
        return out_func


def _instr(
    name: str, in_types: list[IRType], out_types: list[IRType], constant_types: list[Type] = []
) -> IRInstruction:
    return IRInstruction(
        name=name,
        input_dtypes=[t.dtype for t in in_types],
        input_shapes=[t.shape for t in in_types],
        output_dtypes=[t.dtype for t in out_types],
        output_shapes=[t.shape for t in out_types],
        constant_types=constant_types,
    )


class StackInstruction(IRInstruction):
    def __init__(self):
        super().__init__("stack", [], [], [int], [], [])

    def eval_types(self, args: list[ArgType]) -> list[IRType]:
        if len(args) < 2:
            raise ValueError("Provide at least one input to stack and the stack dimension")
        stack_dim = args[-1]
        if not isinstance(stack_dim, int):
            raise ValueError("Must provide stack dimension as last argument")
        arg_type = args[0].type
        if any(arg.type != arg_type for arg in args[1:-1]):
            raise ValueError("All inputs must have the same dtype and shape")
        if stack_dim < 0 or stack_dim > len(arg_type.shape):
            raise ValueError(f"Expected stack dimension between 0 and {len(arg_type.shape)}")
        out_shape = list(arg_type.shape)
        out_shape.insert(stack_dim, len(args) - 1)
        return [IRType(shape=out_shape, dtype=arg_type.dtype)]


class BatchCatInstruction(IRInstruction):
    def __init__(self):
        super().__init__("batch_cat", [], [], [], [], [])

    def eval_types(self, args: list[ArgType]) -> list[IRType]:
        if len(args) < 2:
            raise ValueError("Provide at least one input to stack and the stack dimension")
        if args[-1].type != scalar_int:
            raise ValueError("Expected channel indices as last argument")
        arg_type = args[0].type
        if any(arg.type != arg_type for arg in args[1:-1]):
            raise ValueError("All inputs must have the same dtype and shape")
        return [arg_type]


class BatchSplitInstruction(IRInstruction):
    def __init__(self):
        super().__init__("batch_split", [], [], [int], [], [])

    def eval_types(self, args: list[ArgType]) -> list[IRType]:
        if len(args) != 3:
            raise ValueError("Expected three arguments")
        if args[-2].type != scalar_int:
            raise ValueError("Expected channel indices as second argument")
        n_channels = args[-1]
        if not isinstance(n_channels, int):
            raise ValueError("Expected number of channels as last argument")
        return [args[0].type] * n_channels


def build_instruction_set() -> dict[str, IRInstruction]:
    instructions = [
        _instr("constant", [], [scalar], [float]),
        StackInstruction(),
        BatchCatInstruction(),
        BatchSplitInstruction(),

        # Random numbers
        _instr("uniform", [scalar], [scalar], [float, float]),
        _instr("uniform_inverse", [scalar], [scalar], [float, float]),

        # Math
        _instr("add", [scalar, scalar], [scalar]), #TODO
        _instr("sub", [scalar, scalar], [scalar]), #TODO
        _instr("mul", [scalar, scalar], [scalar]),
        _instr("mul_const", [scalar], [scalar], [float]),
        _instr("clip_min", [scalar], [scalar], [float]), #TODO
        _instr("sqrt", [scalar], [scalar]), #TODO
        _instr("square", [scalar], [scalar]), #TODO

        # Kinematics
        _instr("rotate_zy", [four_vector, scalar, scalar], [four_vector]),
        _instr("boost", [four_vector, four_vector], [four_vector]),
        _instr("boost_inverse", [four_vector, four_vector], [four_vector]),
        IRInstruction(
            name = "boost_beam",
            input_dtypes = [IRDType.float, IRDType.float],
            input_shapes = [("n", 4), ()],
            output_dtypes = [IRDType.float],
            output_shapes = [("n", 4)],
            constant_types = [],
        ),
        IRInstruction(
            name = "boost_beam_inverse",
            input_dtypes = [IRDType.float, IRDType.float],
            input_shapes = [("n", 4), ()],
            output_dtypes = [IRDType.float],
            output_shapes = [("n", 4)],
            constant_types = [],
        ),
        _instr("com_momentum", [scalar], [four_vector]),
        _instr("com_p_in", [scalar], [four_vector, four_vector]), #TODO
        _instr("com_angles", [four_vector], [scalar, scalar]),
        _instr("s", [four_vector], [scalar]),
        _instr("s_and_sqrt_s", [four_vector], [scalar, scalar]),
        _instr("add_4vec", [four_vector, four_vector], [four_vector]),
        _instr("sub_4vec", [four_vector, four_vector], [four_vector]),
        _instr("r_to_x1x2", [scalar, scalar], [scalar, scalar, scalar], [float]),
        _instr("x1x2_to_r", [scalar, scalar], [scalar, scalar], [float]),
        _instr("rapidity", [scalar, scalar], [scalar]), #TODO

        # Two-body decays
        _instr("decay_momentum", [scalar, scalar, scalar, scalar], [four_vector, scalar]),
        _instr("invt_min_max", [scalar, scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("invt_to_costheta", [scalar, scalar, scalar, scalar, scalar, scalar], [scalar]),
        _instr("tinv_two_particle_density", [scalar, scalar, scalar], [scalar]),

        # Invariants
        _instr("uniform_invariant", [scalar, scalar, scalar], [scalar, scalar]),
        _instr("uniform_invariant_inverse", [scalar, scalar, scalar], [scalar, scalar]),
        _instr("breit_wigner_invariant", [scalar, scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("breit_wigner_invariant_inverse", [scalar, scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("stable_invariant", [scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("stable_invariant_inverse", [scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("stable_invariant_nu", [scalar, scalar, scalar, scalar, scalar], [scalar, scalar]),
        _instr("stable_invariant_nu_inverse", [scalar, scalar, scalar, scalar, scalar], [scalar, scalar]),
    ]
    return {instr.name: instr for instr in instructions}

INSTRUCTION_SET = build_instruction_set()
