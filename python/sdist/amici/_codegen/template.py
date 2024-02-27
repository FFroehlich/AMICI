"""Functions to apply template substitution to files."""
from pathlib import Path
from string import Template
from typing import Union


class TemplateAmici(Template):
    """
    Template format used in AMICI (see :class:`string.Template` for more
    details).

    :cvar delimiter:
        delimiter that identifies template variables
    """

    delimiter = "TPL_"


def apply_template(
    source_file: Union[str, Path],
    target_file: Union[str, Path],
    template_data: dict[str, str],
) -> None:
    """
    Load source file, apply template substitution as provided in
    templateData and save as targetFile.

    :param source_file:
        relative or absolute path to template file

    :param target_file:
        relative or absolute path to output file

    :param template_data:
        template keywords to substitute (key is template
        variable without :attr:`TemplateAmici.delimiter`)
    """
    with open(source_file) as filein:
        src = TemplateAmici(filein.read())
    result = src.safe_substitute(template_data)
    with open(target_file, "w") as fileout:
        fileout.write(result)
