"""Import of Antimony models into AMICI.

https://antimony.sourceforge.net/
https://tellurium.readthedocs.io/en/latest/antimony.html
"""
from pathlib import Path
from typing import Union


def antimony2sbml(ant_model: Union[str, Path]) -> str:
    """Convert Antimony model to SBML.

    :param ant_model: Antimony model as string or path to file

    :returns:
        The SBML model as string.
    """
    import antimony as ant

    # Unload everything / free memory
    ant.clearPreviousLoads()
    ant.freeAll()

    try:
        # potentially fails because of too long file name
        is_file = Path(ant_model).exists()
    except OSError:
        is_file = False

    if is_file:
        status = ant.loadAntimonyFile(str(ant_model))
    else:
        status = ant.loadAntimonyString(ant_model)
    if status < 0:
        raise RuntimeError(
            f"Antimony model could not be loaded: {ant.getLastError()}"
        )

    if (main_module_name := ant.getMainModuleName()) is None:
        raise AssertionError("There is no Antimony module.")

    sbml_str = ant.getSBMLString(main_module_name)
    if not sbml_str:
        raise ValueError("Antimony model could not be converted to SBML.")

    return sbml_str


def antimony2amici(ant_model: Union[str, Path], *args, **kwargs):
    """Convert Antimony model to AMICI model.

    Converts the Antimony model provided as string of file to SBML and then imports it into AMICI.

    For documentation see :meth:`amici.sbml_import.SbmlImporter.sbml2amici`.
    """
    from .sbml_import import SbmlImporter

    sbml_str = antimony2sbml(ant_model)
    sbml_importer = SbmlImporter(sbml_str, from_file=False)
    return sbml_importer.sbml2amici(*args, **kwargs)
