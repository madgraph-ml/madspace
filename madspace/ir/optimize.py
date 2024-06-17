from collections import defaultdict
from dataclasses import dataclass
from .ir import IRFunction, IRVariable, IRInstruction, IRInstructionCall


@dataclass
class MergedInstruction:
    instructions: list[IRInstructionCall]
    inputs: set[IRVariable]
    outputs: set[IRVariable]

    def __str__(self):
        return f"[{'; '.join(str(instr) for instr in self.instructions)}]"


class MergeOptimizer:
    def __init__(self, function: IRFunction):
        self.function = function
        self.instructions = list(function.instructions)

    def sort_instructions(self):
        var_ranks = {in_var: 0 for in_var in self.function.inputs}
        instr_ranks = []
        for instr in self.instructions:
            rank = (
                1
                if len(instr.inputs) == 0 else
                max(var_ranks[var] for var in instr.inputs) + 1
            )
            instr_ranks.append(rank)
            var_ranks.update({var: rank for var in instr.outputs})
        instructions, ranks = zip(*sorted(zip(self.instructions, instr_ranks), key=lambda x: x[1]))
        self.instructions = list(instructions)
        self.ranks = list(ranks)

    def dependency_matrix(self):
        self.dep_matrix = []
        var_origins = {}
        for i, instr in enumerate(self.instructions):
            dep = [False] * len(self.instructions)
            for var in instr.inputs:
                if var in var_origins:
                    origin_idx = var_origins[var]
                    dep[origin_idx] = True
                    for dep_idx, dep_val in enumerate(self.dep_matrix[origin_idx]):
                        dep[dep_idx] |= dep_val
                var_origins.update({var: i for var in instr.outputs})
            self.dep_matrix.append(dep)

    def merge_instructions(self):
        for rank_diff in range(self.ranks[-1] - 1):
            for idx1, (instr1, rank1) in enumerate(zip(self.instructions, self.ranks)):
                if instr1 is None:
                    continue
                for idx2, (instr2, rank2) in enumerate(
                    zip(self.instructions[idx1+1:], self.ranks[idx1+1:]),
                    start=idx1+1,
                ):
                    if rank2 - rank1 > rank_diff:
                        break
                    if instr2 is None:
                        continue
                    if self._is_compatible(idx1, idx2):
                        idx2 = self._merge(idx1, idx2)
                        #print(self._check_consistency())

    def _is_compatible(self, idx1: int, idx2: int):
        if self.dep_matrix[idx1][idx2] or self.dep_matrix[idx2][idx1]:
            return False
        instr1, instr2 = self.instructions[idx1], self.instructions[idx2]
        if instr1 is None or instr2 is None:
            return False
        if isinstance(instr1, MergedInstruction):
            instr1 = instr1.instructions[0]
        if isinstance(instr2, MergedInstruction):
            instr2 = instr2.instructions[0]
        if instr1.instruction.name in ["batch_split", "batch_cat"]:
            return False
        if instr1.instruction != instr2.instruction:
            return False
        for const1, const2 in zip(instr1.constants, instr2.constants):
            if const1 != const2:
                return False
        return True

    def _merge(self, idx1: int, idx2: int):
        #instr1, instr2 = self.instructions[idx1], self.instructions[idx2]
        #if not isinstance(instr1, MergedInstruction):
        #    instr1 = MergedInstruction([instr1], set(instr1.inputs), set(instr1.outputs))
        #    self.instructions[idx1] = instr1
        #if isinstance(instr2, MergedInstruction):
        #    instr1.instructions.extend(instr2.instructions)
        #    in2, out2 = instr2.inputs, instr2.outputs
        #else:
        #    instr1.instructions.append(instr2)
        #    in2, out2 = set(instr2.inputs), set(instr2.outputs)
        #instr1.inputs.update(in2)
        #instr1.outputs.update(out2)
        #self.instructions[idx2] = None

        instr1, instr2 = self.instructions[idx1], self.instructions[idx2]
        if not isinstance(instr2, MergedInstruction):
            instr2 = MergedInstruction([instr2], set(instr2.inputs), set(instr2.outputs))
            self.instructions[idx2] = instr2
        if isinstance(instr1, MergedInstruction):
            instr2.instructions.extend(instr1.instructions)
            in1, out1 = instr1.inputs, instr1.outputs
        else:
            instr2.instructions.append(instr1)
            in1, out1 = set(instr1.inputs), set(instr1.outputs)
        instr2.inputs.update(in1)
        instr2.outputs.update(out1)
        self.instructions[idx1] = None

        instr_before = []
        dep_before = []
        index_before = []
        instr_after = []
        dep_after = []
        index_after = []
        for i in range(idx1+1, idx2):
            if self.dep_matrix[i][idx1]:
                instr_after.append(self.instructions[i])
                dep_after.append(self.dep_matrix[i])
                index_after.append(i)
            else:
                instr_before.append(self.instructions[i])
                dep_before.append(self.dep_matrix[i])
                index_before.append(i)
        self.instructions[idx1+1:idx2+1] = instr_before + [instr2] + instr_after
        self.dep_matrix[idx1+1:idx2+1] = dep_before + [self.dep_matrix[idx2]] + dep_after
        for dep in self.dep_matrix[idx1:]:
            dep_new = []
            for i_before in index_before:
                dep_new.append(dep[i_before])
            dep_new.append(dep[idx2])
            for i_after in index_after:
                dep_new.append(dep[i_after])
            dep[idx1+1:idx2+1] = dep_new
        idx2_before = idx2
        idx2 = idx1 + len(instr_before) + 1

        dep1 = list(self.dep_matrix[idx1])
        dep2 = list(self.dep_matrix[idx2])
        dep2[idx2] = True
        for i in range(len(self.instructions)):
            self.dep_matrix[idx1][i] = False
            self.dep_matrix[idx2][i] |= dep1[i]
        for i in range(idx2 + 1, len(self.instructions)):
            if self.dep_matrix[i][idx1]:
                for j in range(len(self.instructions)):
                    self.dep_matrix[i][j] |= dep2[j]
                self.dep_matrix[i][idx1] = False
            if self.dep_matrix[i][idx2]:
                for j in range(len(self.instructions)):
                    self.dep_matrix[i][j] |= dep1[j]

        #if not self._check_consistency():
        #    print(idx1, idx2, idx2_before)
        #    print(instr1, instr2)
        #    raise Exception()

        return idx2

    def _check_consistency(self):
        dep_matrix = []
        var_origins = {}
        known_inputs = set(self.function.inputs)
        for i, (instr, self_dep) in enumerate(zip(self.instructions, self.dep_matrix)):
            dep = [False] * len(self.instructions)
            if instr is None:
                dep_matrix.append(dep)
                continue
            if not all(var in known_inputs for var in instr.inputs):
                print(f"instruction {i} has unknown input")
                return False

            for var in instr.inputs:
                if var in var_origins:
                    origin_idx = var_origins[var]
                    dep[origin_idx] = True
                    for dep_idx, dep_val in enumerate(self.dep_matrix[origin_idx]):
                        dep[dep_idx] |= dep_val
                var_origins.update({var: i for var in instr.outputs})
            dep_matrix.append(dep)
            known_inputs.update(set(instr.outputs))

            for a, b in zip(dep, self_dep):
                if a != b:
                    print(f"inconsistent dep matrix {i}")
                    print(dep[:i+1], self_dep[:i+1], a, b)
                    return False

        #for i in range(len(self.instructions)):
        #    for j in range(len(self.instructions)):
        #        if self.dep_matrix[i][j]:
        #            for k in range(len(self.instructions)):
        #                if self.dep_matrix[j][k] and not self.dep_matrix[i][k]:
        #                    print(f"{i} <- {j} and {j} <- {k} but not {i} <- {k}")
        #                    return False
        return True

    def update_function(self):
        self.function.instructions = self.instructions

    def print(self):
        tmp_count = 0
        def tmp_factory():
            nonlocal tmp_count
            value = f"tmp{tmp_count}"
            tmp_count += 1
            return value
        var_names = defaultdict(tmp_factory)
        var_names.update({in_var: f"input{i}" for i, in_var in enumerate(self.function.inputs)})
        var_names.update({out_var: f"output{i}" for i, out_var in enumerate(self.function.outputs)})
        count = 0
        for instr, rank in zip(self.instructions, self.ranks):
            if instr is None:
                continue
            count += 1
            if isinstance(instr, MergedInstruction):
                print("{")
                for sub_instr in instr.instructions:
                    inputs = ", ".join(
                        [var_names[var] for var in sub_instr.inputs] +
                        [str(const) for const in sub_instr.constants]
                    )
                    output_names = ", ".join(var_names[var] for var in sub_instr.outputs)
                    print(f"    {output_names} = {sub_instr.instruction.name}({inputs}) # {rank}")
                print("}")
            else:
                inputs = ", ".join(
                    [var_names[var] for var in instr.inputs] +
                    [str(const) for const in instr.constants]
                )
                output_names = ", ".join(var_names[var] for var in instr.outputs)
                print(f"{output_names} = {instr.instruction.name}({inputs}) # {rank}")
        print("COUNT", count)
