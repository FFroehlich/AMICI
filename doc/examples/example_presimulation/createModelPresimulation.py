import os

import pysb.export
from pysb.core import (
    Expression,
    Initial,
    Model,
    Monomer,
    Observable,
    Parameter,
    Rule,
)

model = Model()

prot = Monomer("PROT", ["kin", "drug", "phospho"], {"phospho": ["u", "p"]})
prot_0 = Parameter("PROT_0", 10)
Initial(prot(phospho="u", drug=None, kin=None), Expression("initProt", prot_0))

drug = Monomer("DRUG", ["bound"])
drug_0 = Parameter("DRUG_0", 9)
Initial(drug(bound=None), Expression("initDrug", drug_0))

kin = Monomer("KIN", ["bound"])
kin_0 = Parameter("KIN_0", 1)
Initial(kin(bound=None), Expression("initKin", kin_0))

Rule(
    "PROT_DRUG_bind",
    drug(bound=None) + prot(phospho="u", drug=None, kin=None)
    | drug(bound=1) % prot(phospho="u", drug=1, kin=None),
    Parameter("kon_prot_drug", 0.1),
    Parameter("koff_prot_drug", 0.1),
)

Rule(
    "PROT_KIN_bind",
    kin(bound=None) + prot(phospho="u", drug=None, kin=None)
    >> kin(bound=1) % prot(phospho="u", drug=None, kin=1),
    Parameter("kon_prot_kin", 0.1),
)

Rule(
    "PROT_KIN_phospho",
    kin(bound=1) % prot(phospho="u", drug=None, kin=1)
    >> kin(bound=None) + prot(phospho="p", drug=None, kin=None),
    Parameter("kphospho_prot_kin", 0.1),
)

Rule(
    "PROT_dephospho",
    prot(phospho="p", drug=None, kin=None)
    >> prot(phospho="u", drug=None, kin=None),
    Parameter("kdephospho_prot", 0.1),
)

pProt = Observable("pPROT", prot(phospho="p"))
tProt = Observable("tPROT", prot())

Expression("pPROT_obs", pProt / tProt)

sbml_output = pysb.export.export(model, format="sbml")

outfile = os.path.join(
    os.path.dirname(os.path.realpath(__file__)), "model_presimulation.xml"
)
with open(outfile, "w") as f:
    f.write(sbml_output)
