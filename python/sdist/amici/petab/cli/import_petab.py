import argparse

import petab

from ..petab_import import import_model_sbml


def _parse_cli_args():
    """
    Parse command line arguments

    :return:
        Parsed CLI arguments from :mod:`argparse`.
    """
    parser = argparse.ArgumentParser(
        description="Import PEtab-format model into AMICI."
    )

    # General options:
    parser.add_argument(
        "-v",
        "--verbose",
        dest="verbose",
        action="store_true",
        help="More verbose output",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        dest="model_output_dir",
        help="Name of the model directory to create",
    )
    parser.add_argument(
        "--no-compile",
        action="store_false",
        dest="compile",
        help="Only generate model code, do not compile",
    )
    parser.add_argument(
        "--no-validate",
        action="store_false",
        dest="validate",
        help="Skip validation of PEtab files",
    )
    parser.add_argument(
        "--flatten",
        dest="flatten",
        default=False,
        action="store_true",
        help="Flatten measurement specific overrides of "
        "observable and noise parameters",
    )
    parser.add_argument(
        "--no-sensitivities",
        dest="generate_sensitivity_code",
        default=True,
        action="store_false",
        help="Skip generation of sensitivity code",
    )

    # Call with set of files
    parser.add_argument(
        "-s", "--sbml", dest="sbml_file_name", help="SBML model filename"
    )
    parser.add_argument(
        "-m",
        "--measurements",
        dest="measurement_file_name",
        help="Measurement table",
    )
    parser.add_argument(
        "-c",
        "--conditions",
        dest="condition_file_name",
        help="Conditions table",
    )
    parser.add_argument(
        "-p",
        "--parameters",
        dest="parameter_file_name",
        help="Parameter table",
    )
    parser.add_argument(
        "-b",
        "--observables",
        dest="observable_file_name",
        help="Observable table",
    )

    parser.add_argument(
        "-y",
        "--yaml",
        dest="yaml_file_name",
        help="PEtab YAML problem filename",
    )

    parser.add_argument(
        "-n",
        "--model-name",
        dest="model_name",
        help="Name of the python module generated for the " "model",
    )

    args = parser.parse_args()

    if not args.yaml_file_name and not all(
        (
            args.sbml_file_name,
            args.condition_file_name,
            args.observable_file_name,
        )
    ):
        parser.error(
            "When not specifying a model name or YAML file, then "
            "SBML, condition and observable file must be specified"
        )

    return args


def _main():
    """
    Command line interface to import a model in the PEtab
    (https://github.com/PEtab-dev/PEtab/) format into AMICI.
    """
    args = _parse_cli_args()

    if args.yaml_file_name:
        pp = petab.Problem.from_yaml(args.yaml_file_name)
    else:
        pp = petab.Problem.from_files(
            sbml_file=args.sbml_file_name,
            condition_file=args.condition_file_name,
            measurement_file=args.measurement_file_name,
            parameter_file=args.parameter_file_name,
            observable_files=args.observable_file_name,
        )

    # Check for valid PEtab before potentially modifying it
    if args.validate:
        petab.lint_problem(pp)

    if args.flatten:
        petab.flatten_timepoint_specific_output_overrides(pp)

    import_model_sbml(
        model_name=args.model_name,
        sbml_model=pp.sbml_model,
        condition_table=pp.condition_df,
        observable_table=pp.observable_df,
        measurement_table=pp.measurement_df,
        model_output_dir=args.model_output_dir,
        compile=args.compile,
        generate_sensitivity_code=args.generate_sensitivity_code,
        verbose=args.verbose,
        validate=False,
    )


if __name__ == "__main__":
    _main()
