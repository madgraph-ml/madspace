from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from collections import defaultdict


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
four_vector = IRType(shape=(4,))


@dataclass(eq=False)
class IRVariable:
    type: IRType

    def __hash__(self):
        return id(self)


@dataclass
class IRInstruction:
    name: str
    input_dtypes: list[IRDType, ...]
    input_shapes: list[tuple[int | str, ...], ...]
    output_dtypes: list[IRDType, ...]
    output_shapes: list[tuple[int | str, ...], ...]

    def eval_types(self, args: list[IRVariable]) -> list[IRType]:
        assert len(self.input_dtypes) == len(self.input_shapes)
        assert len(self.output_dtypes) == len(self.output_shapes)

        variables = {}
        for i, (input_dtype, input_shape, arg) in enumerate(
            zip(self.input_dtypes, self.input_shapes, args)
        ):
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
        return IRInstructionCall(
            instruction = self,
            inputs = args,
            outputs = [IRVariable(output_type) for output_type in self.eval_types(args)],
        )


@dataclass
class IRInstructionCall:
    instruction: IRInstruction
    inputs: list[IRVariable]
    outputs: list[IRVariable]


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
        def add_instr(*args):
            instr = instruction(*args)
            self.instructions.append(instr)
            return instr.outputs
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
            input_names = ", ".join(var_names[var] for var in instr.inputs)
            output_names = ", ".join(var_names[var] for var in instr.outputs)
            lines.append(f"{output_names} = {instr.instruction.name}({input_names})")
        return "\n".join(lines)

    def _check_outputs(self):
        for i, (output, output_type) in enumerate(zip(self.outputs, self.output_types)):
            if output is None:
                raise ValueError(f"No value for output {i}")
            if output.type != output_type:
                raise ValueError(f"Output {i}: Expected type {output_type}, got {output.type}")


    def optimize(self):
        pass

def _instr(name: str, in_types: list[IRType], out_types: list[IRType]) -> IRInstruction:
    return IRInstruction(
        name=name,
        input_dtypes=[t.dtype for t in in_types],
        input_shapes=[t.shape for t in in_types],
        output_dtypes=[t.dtype for t in out_types],
        output_shapes=[t.shape for t in out_types],
    )

def build_instruction_set() -> dict[str, IRInstruction]:
    instructions = [
        # Random numbers
        _instr("unit_to_phi", [scalar], [scalar]),
        _instr("unit_to_costheta", [scalar], [scalar]),

        # Kinematics
        _instr("rotate_zy", [four_vector, scalar, scalar], [four_vector]),
        _instr("boost", [four_vector, four_vector], [four_vector]),
        _instr("boost_inverse", [four_vector, four_vector], [four_vector]),
        _instr("com_momentum", [scalar], [four_vector]),
        _instr("com_angles", [four_vector], [scalar, scalar]),
        _instr("s", [four_vector], [scalar]),
        _instr("sqrt_s_and_s", [four_vector], [scalar, scalar]),
        _instr("add_4vec", [four_vector, four_vector], [four_vector]),
        _instr("sub_4vec", [four_vector, four_vector], [four_vector]),

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
