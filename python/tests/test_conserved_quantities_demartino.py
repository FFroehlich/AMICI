"""Tests for conservation laws / conserved moieties"""
import os
from time import perf_counter

import numpy as np
import pytest
import sympy as sp
from amici.conserved_quantities_demartino import _fill, _kernel
from amici.conserved_quantities_demartino import _output as output
from amici.conserved_quantities_demartino import (
    compute_moiety_conservation_laws,
)
from amici.logging import get_logger, log_execution_time
from amici.testing import skip_on_valgrind

logger = get_logger(__name__)

# reference data for `engaged_species` after kernel()
demartino2014_kernel_engaged_species = [
    179,
    181,
    185,
    186,
    187,
    190,
    191,
    194,
    195,
    197,
    198,
    200,
    208,
    209,
    210,
    211,
    214,
    215,
    218,
    219,
    221,
    222,
    224,
    277,
    292,
    340,
    422,
    467,
    468,
    490,
    491,
    598,
    613,
    966,
    968,
    1074,
    1171,
    1221,
    1223,
    1234,
    1266,
    1478,
    1479,
    1480,
    1481,
    1496,
    1497,
    1498,
    1501,
    1526,
    1527,
    1528,
    1529,
    394,
    1066,
    398,
    465,
    466,
    594,
    671,
    429,
    990,
    652,
    655,
    662,
    663,
    664,
    665,
    666,
    667,
    668,
    669,
    759,
    760,
    920,
    921,
    569,
    1491,
    1055,
    1546,
    276,
    1333,
    1421,
    1429,
    1430,
    1438,
    1551,
    1428,
    1439,
    1552,
    1513,
    1553,
    1520,
    1523,
    1530,
    1531,
    384,
    1536,
    440,
    1537,
    447,
    1538,
    456,
    1539,
    582,
    1540,
    876,
    1541,
    885,
    1542,
    911,
    1543,
    978,
    1544,
    1010,
    1545,
    1070,
    1547,
    761,
    1127,
    1548,
    1324,
    1549,
    1370,
    1550,
    1554,
    1560,
    1555,
    1580,
    1556,
    1644,
]


@pytest.fixture(scope="session")
def data_demartino2014():
    """Get tests from DeMartino2014 Suppl. Material"""
    import gzip
    import pooch

    # stoichiometric matrix
    data = gzip.GzipFile(
        pooch.retrieve(
            "https://github.com/AMICI-dev/AMICI/files/11430971/DeMartinoDe2014_test-ecoli.dat.gz",
            known_hash="md5:899873f8f1c413d13c3f8e94c1496b7e",
        )
    )
    S = [
        int(item)
        for sl in [
            entry.decode("ascii").strip().split("\t")
            for entry in data.readlines()
        ]
        for item in sl
    ]

    # metabolite / row names
    with open(
        pooch.retrieve(
            "https://github.com/AMICI-dev/AMICI/files/11430970/test-ecoli-met.txt",
            known_hash="md5:d71e711a3655311390b38d00dcd6aa7f",
        )
    ) as f:
        row_names = [entry.strip() for entry in f.readlines()]
    return S, row_names


@skip_on_valgrind
def test_kernel_demartino2014(data_demartino2014, quiet=True):
    """Invoke test case and benchmarking for De Martino's published results
    for E. coli network. Kernel-only."""
    stoichiometric_list, row_names = data_demartino2014
    num_species = 1668
    num_reactions = 2381
    assert (
        len(stoichiometric_list) == num_species * num_reactions
    ), "Unexpected dimension of stoichiometric matrix"

    # Expected number of metabolites per conservation law found after kernel()
    expected_num_species = (
        [53] + [2] * 11 + [6] + [3] * 2 + [2] * 15 + [3] + [2] * 5
    )

    (
        kernel_dim,
        engaged_species,
        int_kernel_dim,
        conserved_moieties,
        cls_species_idxs,
        cls_coefficients,
    ) = _kernel(stoichiometric_list, num_species, num_reactions)

    if not quiet:
        output(
            int_kernel_dim,
            kernel_dim,
            engaged_species,
            cls_species_idxs,
            cls_coefficients,
            row_names,
        )

    # There are 38 conservation laws, engaging 131 metabolites
    # 36 are integers (conserved moieties), engaging 128 metabolites (from C++)
    assert kernel_dim == 38, "Not all conservation laws found"
    assert int_kernel_dim == 36, "Not all conserved moiety laws found"
    assert (
        engaged_species == demartino2014_kernel_engaged_species
    ), "Wrong engaged metabolites reported"
    assert (
        len(conserved_moieties) == 128
    ), "Wrong number of conserved moieties reported"

    # Assert that each conserved moiety has the correct number of metabolites
    for i in range(int_kernel_dim - 2):
        assert (
            len(cls_species_idxs[i]) == expected_num_species[i]
        ), f"Moiety #{i + 1} failed for test case (De Martino et al.)"


@skip_on_valgrind
def test_fill_demartino2014(data_demartino2014):
    """Test creation of interaction matrix"""
    stoichiometric_list, row_names = data_demartino2014
    num_species = 1668
    J, J2, fields = _fill(
        stoichiometric_list, demartino2014_kernel_engaged_species, num_species
    )
    ref_for_J = [
        [25, 27],
        [12, 42],
        [13, 43],
        [14, 44],
        [15, 41],
        [16, 45],
        [17, 47],
        [18, 48],
        [19, 23, 49],
        [20, 50],
        [21, 51],
        [22, 52],
        [1, 23, 30, 35],
        [2, 23, 29, 35],
        [3, 23, 35, 46],
        [4, 23, 33, 35],
        [5, 23, 31, 35],
        [6, 23, 35, 37],
        [7, 23, 28, 35],
        [8, 23, 32, 35],
        [9, 23, 34, 35],
        [10, 23, 35, 40],
        [11, 23, 35, 36],
        [
            8,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            24,
            25,
            26,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            46,
        ],
        [23, 25],
        [0, 23, 24, 35],
        [23],
        [0, 28],
        [18, 23, 27, 35],
        [13, 23, 35, 42],
        [12, 23, 35, 47],
        [16, 23, 35, 47],
        [19, 23, 35, 45],
        [15, 23, 35, 44],
        [20, 23, 35, 48],
        [
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            25,
            28,
            29,
            30,
            31,
            32,
            33,
            34,
            36,
            37,
            40,
            46,
        ],
        [22, 23, 35, 49],
        [17, 23, 35, 50],
        [23, 51],
        [23, 41],
        [21, 23, 35, 52],
        [4, 39],
        [1, 29],
        [2, 46],
        [3, 33],
        [5, 32],
        [14, 23, 35, 43],
        [6, 30, 31],
        [7, 34],
        [8, 36],
        [9, 37],
        [10, 38],
        [11, 40],
        [54],
        [53],
        [58, 80],
        [57, 59, 82],
        [56],
        [55, 59, 80],
        [56, 58, 82],
        [61],
        [60],
        [63],
        [62],
        [65],
        [64],
        [67, 68, 69],
        [66, 68, 69],
        [66, 67, 69, 70, 71, 94, 95],
        [66, 67, 68, 70, 71, 94, 95],
        [68, 69, 71],
        [68, 69, 70],
        [73],
        [72],
        [75],
        [74],
        [77],
        [76],
        [79],
        [78],
        [55, 58, 81],
        [80],
        [56, 59],
        [84],
        [83, 85, 87],
        [84, 86, 87],
        [85],
        [84, 85],
        [89],
        [88],
        [91],
        [90],
        [93],
        [92],
        [68, 69, 95],
        [68, 69, 94],
        [97],
        [96],
        [99],
        [98],
        [101],
        [100],
        [103],
        [102],
        [105],
        [104],
        [107],
        [106],
        [109],
        [108],
        [111],
        [110],
        [113],
        [112],
        [115],
        [114],
        [117],
        [116],
        [119],
        [118, 120],
        [119],
        [122],
        [121],
        [124],
        [123],
        [126],
        [125],
        [128],
        [127],
        [130],
        [129],
    ]
    ref_for_J2 = [
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, -2, -1],
        [-1, -1],
        [-1, -1],
        [-1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [-1, 1, -1, -1],
        [
            -2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            -2,
            1,
            -1,
            -1,
            -1,
            -1,
            -3,
            -6,
            -6,
            -1,
            -13,
            -7,
            -3,
            -3,
            -3,
            -5,
            -5,
        ],
        [-2, -1],
        [-1, 1, -1, -2],
        [-1],
        [-1, -2],
        [-1, -1, -2, 1],
        [-1, -1, 1, -2],
        [-1, -1, 1, -1],
        [-1, -3, 1, -2],
        [-1, -6, 1, -2],
        [-1, -6, 1, -2],
        [-1, -1, 1, -2],
        [
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -1,
            -13,
            -2,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
        ],
        [-1, -7, 1, -2],
        [-1, -3, 1, -2],
        [-3, -2],
        [-3, -2],
        [-1, -5, 1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -5, 1, -2],
        [-1, -1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-1, -2],
        [-2],
        [-2],
        [1, -1],
        [-2, -1, -1],
        [-2],
        [1, -1, -1],
        [-1, -1, 1],
        [-1],
        [-1],
        [-2],
        [-2],
        [-2],
        [-2],
        [-2, -1, 1],
        [-2, 1, -1],
        [-1, 1, -3, -1, 1, -1, 1],
        [1, -1, -3, 1, -1, 1, -1],
        [-1, 1, -2],
        [1, -1, -2],
        [-5],
        [-5],
        [-6],
        [-6],
        [-2],
        [-2],
        [-1],
        [-1],
        [-1, -1, -1],
        [-1],
        [-1, 1],
        [-1],
        [-1, 1, -1],
        [1, -1, -1],
        [-1],
        [-1, -1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-2],
        [-2],
        [-1, 1, -10],
        [1, -1, -10],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-2],
        [-2],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1, -1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
        [-1],
    ]
    ref_for_fields = [
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        4,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        51,
        3,
        3,
        1,
        3,
        3,
        3,
        2,
        5,
        8,
        8,
        3,
        15,
        9,
        5,
        5,
        5,
        7,
        3,
        3,
        3,
        3,
        3,
        7,
        4,
        3,
        3,
        3,
        3,
        3,
        2,
        2,
        1,
        3,
        2,
        2,
        2,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        3,
        3,
        2,
        2,
        5,
        5,
        6,
        6,
        2,
        2,
        1,
        1,
        2,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        10,
        10,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
    ]
    # compare J from Python with reference from C++
    for i in range(len(ref_for_J)):
        assert (
            J[i] == ref_for_J[i]
        ), f"J_{i} ({J[i]}) does not match J_{i}_ref ({ref_for_J[i]})"
    assert not any(J[len(ref_for_J) :])

    # compare J2 from Python with reference from C++
    for i in range(len(ref_for_J2)):
        assert (
            J2[i] == ref_for_J2[i]
        ), f"J_{i} ({J2[i]}) does not match J_{i}_ref ({ref_for_J2[i]})"
    assert not any(J2[len(ref_for_J2) :])

    # compare fields from Python with reference from C++
    for i in range(len(ref_for_fields)):
        assert (
            fields[i] == ref_for_fields[i]
        ), f"J_{i} ({fields[i]}) does not match J_{i}_ref ({ref_for_fields[i]})"
    assert not any(fields[len(ref_for_fields) :])


def compute_moiety_conservation_laws_demartino2014(
    data_demartino2014, quiet=False
):
    """Compute conserved quantities for De Martino's published results
    for E. coli network"""
    stoichiometric_list, row_names = data_demartino2014

    num_species = 1668
    num_reactions = 2381
    assert (
        len(stoichiometric_list) == num_species * num_reactions
    ), "Unexpected dimension of stoichiometric matrix"

    start = perf_counter()
    cls_state_idxs, cls_coefficients = compute_moiety_conservation_laws(
        stoichiometric_list,
        num_species=num_species,
        num_reactions=num_reactions,
    )
    runtime = perf_counter() - start
    if not quiet:
        print(f"Execution time: {runtime} [s]")

    assert len(cls_state_idxs) == len(cls_coefficients) == 38
    return runtime


@skip_on_valgrind
def test_compute_moiety_conservation_laws_demartino2014(data_demartino2014):
    """Invoke test case and benchmarking for De Martino's published results
    for E. coli network"""
    compute_moiety_conservation_laws_demartino2014(
        data_demartino2014, quiet=False
    )


@skip_on_valgrind
@log_execution_time("Detecting moiety conservation laws", logger)
def test_cl_detect_execution_time(data_demartino2014):
    """Test execution time stays within a certain predefined bound.
    As the algorithm is non-deterministic, allow for some retries.
    Only one has to succeed."""
    max_tries = 5
    # <5s on modern hardware, but leave some slack
    max_time_seconds = 40 if "GITHUB_ACTIONS" in os.environ else 10

    runtime = np.Inf

    for _ in range(max_tries):
        runtime = compute_moiety_conservation_laws_demartino2014(
            data_demartino2014, quiet=True
        )
        if runtime < max_time_seconds:
            break
    assert runtime < max_time_seconds, "Took too long"


@skip_on_valgrind
def test_compute_moiety_conservation_laws_simple():
    """Test a simple example, ensure the conservation laws are identified
    reliably. Requires the Monte Carlo to identify all."""
    stoichiometric_matrix = sp.Matrix(
        [[-1.0, 1.0], [-1.0, 1.0], [1.0, -1.0], [1.0, -1.0]]
    )
    stoichiometric_list = [
        float(entry) for entry in stoichiometric_matrix.T.flat()
    ]

    num_tries = 1000
    found_all_n_times = 0
    for _ in range(num_tries):
        cls_state_idxs, cls_coefficients = compute_moiety_conservation_laws(
            stoichiometric_list, *stoichiometric_matrix.shape
        )

        assert cls_state_idxs in (
            [[0, 3], [1, 2], [1, 3]],
            [[0, 3], [1, 2], [0, 2]],
            # should happen rarely
            [[0, 3], [1, 2]],
        )
        assert cls_coefficients in (
            [[1.0, 1.0], [1.0, 1.0], [1.0, 1.0]],
            [[1.0, 1.0], [1.0, 1.0]],
        )

        num_cls_found = len(cls_state_idxs)
        if num_cls_found == 3:
            found_all_n_times += 1
    # sometimes we don't find all conservation laws, but this should be rare
    assert found_all_n_times / num_tries >= 0.995
