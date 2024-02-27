"""Functions related to SWIG or SWIG-generated code"""
import ast
import contextlib
import re


class TypeHintFixer(ast.NodeTransformer):
    """Replaces SWIG-generated C++ typehints by corresponding Python types"""

    mapping = {
        "void": None,
        "double": ast.Name("float"),
        "int": ast.Name("int"),
        "long": ast.Name("int"),
        "ptrdiff_t": ast.Name("int"),
        "size_t": ast.Name("int"),
        "bool": ast.Name("bool"),
        "std::unique_ptr< amici::Solver >": ast.Constant("Solver"),
        "amici::InternalSensitivityMethod": ast.Constant(
            "InternalSensitivityMethod"
        ),
        "amici::InterpolationType": ast.Constant("InterpolationType"),
        "amici::LinearMultistepMethod": ast.Constant("LinearMultistepMethod"),
        "amici::LinearSolver": ast.Constant("LinearSolver"),
        "amici::Model *": ast.Constant("Model"),
        "amici::Model const *": ast.Constant("Model"),
        "amici::NewtonDampingFactorMode": ast.Constant(
            "NewtonDampingFactorMode"
        ),
        "amici::NonlinearSolverIteration": ast.Constant(
            "NonlinearSolverIteration"
        ),
        "amici::ObservableScaling": ast.Constant("ObservableScaling"),
        "amici::ParameterScaling": ast.Constant("ParameterScaling"),
        "amici::RDataReporting": ast.Constant("RDataReporting"),
        "amici::SensitivityMethod": ast.Constant("SensitivityMethod"),
        "amici::SensitivityOrder": ast.Constant("SensitivityOrder"),
        "amici::Solver *": ast.Constant("Solver"),
        "amici::SteadyStateSensitivityMode": ast.Constant(
            "SteadyStateSensitivityMode"
        ),
        "amici::realtype": ast.Name("float"),
        "DoubleVector": ast.Constant("Sequence[float]"),
        "IntVector": ast.Name("Sequence[int]"),
        "std::string": ast.Name("str"),
        "std::string const &": ast.Name("str"),
        "std::unique_ptr< amici::ExpData >": ast.Constant("ExpData"),
        "std::unique_ptr< amici::ReturnData >": ast.Constant("ReturnData"),
        "std::vector< amici::ParameterScaling,"
        "std::allocator< amici::ParameterScaling > > const &": ast.Constant(
            "ParameterScalingVector"
        ),
    }

    def visit_FunctionDef(self, node):
        # Has a return type annotation?
        if node.returns:
            node.returns = self._new_annot(node.returns.value)

        # Has arguments?
        if node.args.args:
            for arg in node.args.args:
                if not arg.annotation:
                    continue
                if isinstance(arg.annotation, ast.Name):
                    # there is already proper annotation
                    continue

                arg.annotation = self._new_annot(arg.annotation.value)
        return node

    def _new_annot(self, old_annot: str):
        with contextlib.suppress(KeyError):
            return self.mapping[old_annot]

        # std::vector size type
        if re.match(r"std::vector< .* >::(?:size|difference)_type", old_annot):
            return ast.Name("int")

        # std::vector value type
        if (
            value_type := re.sub(
                r"std::vector< (.*) >::value_type(?: const &)?",
                r"\1",
                old_annot,
            )
        ) in self.mapping:
            return self.mapping[value_type]

        # std::vector
        if (
            value_type := re.sub(
                r"std::vector< (.*),std::allocator< \1 > >(?: const &)?",
                r"\1",
                old_annot,
            )
        ) in self.mapping:
            value_type_annot = self.mapping[value_type]
            if isinstance(value_type_annot, ast.Constant):
                return ast.Name(f"tuple['{value_type_annot.value}']")
            if isinstance(value_type_annot, ast.Name):
                return ast.Name(f"tuple[{value_type_annot.id}]")

        return ast.Constant(old_annot)


def fix_typehints(infilename, outfilename):
    """Change SWIG-generated C++ typehints to Python typehints"""
    # Only available from Python3.9
    if not getattr(ast, "unparse", None):
        return

    # file -> AST
    with open(infilename) as f:
        source = f.read()
    parsed_source = ast.parse(source)

    # Change AST
    fixer = TypeHintFixer()
    parsed_source = fixer.visit(parsed_source)

    # AST -> file
    with open(outfilename, "w") as f:
        f.write(ast.unparse(parsed_source))
