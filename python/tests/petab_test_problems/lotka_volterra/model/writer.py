from pathlib import Path

import yaml2sbml

yaml2sbml_yaml = "lotka_volterra.yaml"
petab_path = Path(__file__).parent.parent / "petab"
petab_yaml = "problem.yaml"
measurements_tsv = "measurements.tsv"
model_name = "lotka_volterra"

yaml2sbml.yaml2petab(
    yaml_dir=yaml2sbml_yaml,
    output_dir=str(petab_path),
    sbml_name=model_name,
    petab_yaml_name=petab_yaml,
    measurement_table_name=measurements_tsv,
)
