#
# Configuration file for the Sphinx documentation builder.
#
# This file does only contain a selection of the most common options. For a
# full list see the documentation:
# http://www.sphinx-doc.org/en/stable/config
import os
import re
import subprocess
import sys
from enum import EnumType

# need to import before setting typing.TYPE_CHECKING=True, fails otherwise
import amici
import exhale.deploy
import exhale_multiproject_monkeypatch
from unittest import mock
import pandas as pd
import sphinx
import sympy as sp
from exhale import configs as exhale_configs
from sphinx.transforms.post_transforms import ReferencesResolver

exhale_multiproject_monkeypatch, pd, sp  # to avoid removal of unused import

# BEGIN Monkeypatch exhale
from exhale.deploy import _generate_doxygen as exhale_generate_doxygen


def my_exhale_generate_doxygen(doxygen_input):
    """Monkey-patch exhale for post-processing doxygen output"""

    # run mtocpp_post
    doxy_xml_dir = exhale_configs._doxygen_xml_output_directory
    if "matlab" in doxy_xml_dir:
        print("Running mtocpp_post on ", doxy_xml_dir)
        mtocpp_post = os.path.join(
            amici_dir, "ThirdParty", "mtocpp-master", "build", "mtocpp_post"
        )
        subprocess.run([mtocpp_post, doxy_xml_dir])

    # let exhale do its job
    exhale_generate_doxygen(doxygen_input)


exhale.deploy._generate_doxygen = my_exhale_generate_doxygen
# END Monkeypatch exhale


# BEGIN Monkeypatch breathe
from breathe.renderer.sphinxrenderer import (
    DomainDirectiveFactory as breathe_DomainDirectiveFactory,
)

old_breathe_DomainDirectiveFactory_create = (
    breathe_DomainDirectiveFactory.create
)


def my_breathe_DomainDirectiveFactory_create(domain: str, args):
    if domain != "mat":
        return old_breathe_DomainDirectiveFactory_create(domain, args)

    from sphinxcontrib.matlab import MatClassmember, MATLABDomain

    matlab_classes = {k: (v, k) for k, v in MATLABDomain.directives.items()}
    matlab_classes["variable"] = (MatClassmember, "attribute")
    cls, name = matlab_classes[args[0]]
    return cls(domain + ":" + name, *args[1:])


breathe_DomainDirectiveFactory.create = (
    my_breathe_DomainDirectiveFactory_create
)


# END Monkeypatch breathe


def install_mtocpp():
    """Install mtocpp (Matlab doxygen filter)"""
    cmd = os.path.join(amici_dir, "scripts", "downloadAndBuildMtocpp.sh")
    ret = subprocess.run(
        cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
    )
    if ret.returncode != 0:
        print(ret.stdout.decode("utf-8"))
        raise RuntimeError("downloadAndBuildMtocpp.sh failed")


def install_doxygen():
    """Get a more recent doxygen"""
    version = "1.9.7"
    doxygen_exe = os.path.join(
        amici_dir, "ThirdParty", f"doxygen-{version}", "bin", "doxygen"
    )
    # to create a symlink to doxygen in a location that is already on PATH
    some_dir_on_path = os.environ["PATH"].split(os.pathsep)[0]
    cmd = (
        f"cd '{os.path.join(amici_dir, 'ThirdParty')}' "
        f"&& wget 'https://www.doxygen.nl/files/"
        f"doxygen-{version}.linux.bin.tar.gz' "
        f"&& tar -xzf doxygen-{version}.linux.bin.tar.gz "
        f"&& ln -sf '{doxygen_exe}' '{some_dir_on_path}'"
    )
    subprocess.run(cmd, shell=True, check=True)
    assert os.path.islink(os.path.join(some_dir_on_path, "doxygen"))
    # verify it's available
    res = subprocess.run(
        ["doxygen", "--version"], check=False, capture_output=True
    )
    print(res.stdout.decode(), res.stderr.decode())
    assert version in res.stdout.decode()


# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

amici_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# -- RTD custom build --------------------------------------------------------

# only execute those commands when running from RTD
if "READTHEDOCS" in os.environ and os.environ["READTHEDOCS"]:
    install_doxygen()

# Required for matlab doxygen processing
install_mtocpp()


# -- Project information -----------------------------------------------------
# The short X.Y version
version = amici.__version__
# The full version, including alpha/beta/rc tags
release = version

project = "AMICI"
copyright = "2020, The AMICI developers"
author = "The AMICI developers"
title = "AMICI Documentation"

# -- Mock out some problematic modules-------------------------------------

# Note that for sub-modules, all parent modules must be listed explicitly.
autodoc_mock_imports = ["_amici", "amici._amici"]
for mod_name in autodoc_mock_imports:
    sys.modules[mod_name] = mock.MagicMock()

# -- General configuration ---------------------------------------------------

# If your documentation needs a minimal Sphinx version, state it here.
#
# needs_sphinx = '1.0'

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "readthedocs_ext.readthedocs",
    # Required, e.g. for PEtab-derived classes where the base class has non-rst
    #  docstrings
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.coverage",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "sphinxcontrib.matlab",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "recommonmark",
    "sphinx_autodoc_typehints",
    "hoverxref.extension",
    "breathe",
    "exhale",
]

intersphinx_mapping = {
    "pysb": ("https://pysb.readthedocs.io/en/stable/", None),
    "petab": (
        "https://petab.readthedocs.io/projects/libpetab-python/en/latest/",
        None,
    ),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
    "python": ("https://docs.python.org/3", None),
}

# Add notebooks prolog with binder links
# get current git reference
ret = subprocess.run("git rev-parse HEAD".split(" "), capture_output=True)
ref = ret.stdout.rstrip().decode()
nbsphinx_prolog = (
    f"{{% set {ref=} %}}"
    r"""
    {% set docname = "documentation/" + env.doc2path(env.docname, base=False) %}
    .. raw:: html

        <div class="note">
          <a href="https://mybinder.org/v2/gh/AMICI-dev/AMICI/{{ ref|e }}?labpath={{ docname|e }}" target="_blank">
          <img src="https://mybinder.org/badge_logo.svg" alt="Open in binder"/></a>
        </div>

    """
)

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = [".rst", ".md"]

# The master toctree document.
master_doc = "index"

# The language for content autogenerated by Sphinx. Refer to documentation
# for a list of supported languages.
#
# This is also used if you do content translation via gettext catalogs.
# Usually you set "language" from the command line for these cases.
language = "en"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path .
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "**.ipynb_checkpoints",
    "numpy.py",
    "INSTALL.md",
    "MATLAB_.md",
    "CPP_.md",
    "gfx",
]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# autodoc
autodoc_default_options = {
    "special-members": "__init__",
    "inherited-members": True,
    "undoc-members": True,
}

# sphinx-autodoc-typehints
typehints_fully_qualified = True
typehints_document_rtype = True
set_type_checking_flag = True

# hoverxref
hoverxref_auto_ref = True
hoverxref_roles = ["term"]
hoverxref_domains = ["py"]
hoverxref_role_types = {
    "hoverxref": "tooltip",
    "ref": "tooltip",
    "term": "tooltip",
    "obj": "tooltip",
    "func": "tooltip",
    "mod": "tooltip",
    "meth": "tooltip",
    "class": "tooltip",
}

# breathe settings
breathe_projects = {
    "AMICI_Matlab": "./_doxyoutput_amici_matlab/xml",
    "AMICI_CPP": "./_doxyoutput_amici_cpp/xml",
}

breathe_default_project = "AMICI_CPP"
breathe_domain_by_extension = {
    "m": "mat",
    "h": "cpp",
    "cpp": "cpp",
}

# exhale settings
exhale_args = {
    "rootFileName": "library_root.rst",
    "doxygenStripFromPath": "..",
    "createTreeView": True,
    # TIP: if using the sphinx-bootstrap-theme, you need
    # "treeViewIsBootstrap": True,
    "exhaleExecutesDoxygen": True,
    "verboseBuild": True,
}

mtocpp_filter = os.path.join(
    amici_dir, "matlab", "mtoc", "config", "mtocpp_filter.sh"
)
exhale_projects_args = {
    "AMICI_CPP": {
        "exhaleDoxygenStdin": "\n".join(
            [
                "INPUT = ../include/amici",
                "BUILTIN_STL_SUPPORT    = YES",
                "PREDEFINED            += EXHALE_DOXYGEN_SHOULD_SKIP_THIS",
                "EXCLUDE += ../include/amici/interface_matlab.h",
                "EXCLUDE += ../include/amici/returndata_matlab.h",
                "EXCLUDE += ../include/amici/spline.h",
                # amici::log collides with amici::${some_enum}::log
                #  potentially fixed in
                #  https://github.com/svenevs/exhale/commit/c924df2e139a09fbacd07587779c55fd0ee4e00b
                #  and can be un-excluded after the next exhale release
                "EXCLUDE += ../include/amici/symbolic_functions.h",
            ]
        ),
        "containmentFolder": "_exhale_cpp_api",
        "rootFileTitle": "AMICI C++ API",
        "afterTitleDescription": "AMICI C++ library functions",
    },
    # Third Party Project Includes
    "AMICI_Matlab": {
        "exhaleDoxygenStdin": "\n".join(
            [
                "INPUT = ../matlab",
                "EXTENSION_MAPPING = .m=C++",
                "FILTER_PATTERNS = " f"*.m={mtocpp_filter}",
                "EXCLUDE += ../matlab/examples",
                "EXCLUDE += ../matlab/mtoc",
                "EXCLUDE += ../matlab/SBMLimporter",
                "EXCLUDE += ../matlab/auxiliary",
                "EXCLUDE += ../matlab/tests",
                "PREDEFINED += EXHALE_DOXYGEN_SHOULD_SKIP_THIS",
            ]
        ),
        "containmentFolder": "_exhale_matlab_api",
        "rootFileTitle": "AMICI Matlab API",
        "afterTitleDescription": "AMICI Matlab library functions",
        "lexerMapping": {r".*\.m$": "matlab"},
    },
}
# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = "sphinx_rtd_theme"

# Theme options are theme-specific and customize the look and feel of a theme
# further.  For a list of options available for each theme, see the
# documentation.
#
# html_theme_options = {}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
# html_static_path = ['_static']
html_favicon = "gfx/logo.png"

# Custom sidebar templates, must be a dictionary that maps document names
# to template names.
#
# The default sidebars (for documents that don't match any pattern) are
# defined by theme itself.  Builtin themes are using these templates by
# default: ``['localtoc.html', 'relations.html', 'sourcelink.html',
# 'searchbox.html']``.
#
# html_sidebars = {}


# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "AMICIdoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "AMICI.tex", title, author, "manual"),
]

# -- Options for manual page output ------------------------------------------

# One entry per manual page. List of tuples
# (source start file, name, description, authors, manual section).
man_pages = [(master_doc, "amici", title, [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "AMICI",
        title,
        author,
        "AMICI",
        "Advanced Multilanguage Interface for CVODES and IDAS.",
        "Miscellaneous",
    ),
]

# Custom processing routines for docstrings and signatures

typemaps = {
    "std::vector< amici::realtype,std::allocator< amici::realtype > >": "DoubleVector",
    "std::vector< double,std::allocator< double > >": "DoubleVector",
    "std::vector< int,std::allocator< int > >": "IntVector",
    "std::vector< amici::ParameterScaling,std::allocator< "
    "amici::ParameterScaling >": "ParameterScalingVector",
    "std::vector< std::string,std::allocator< std::string > >": "StringVector",
    "std::vector< bool,std::allocator< bool > >": "BoolVector",
    "std::map< std::string,amici::realtype,std::less< std::string >,"
    "std::allocator< std::pair< std::string const,amici::realtype > > >": "StringDoubleMap",
    "std::vector< amici::ExpData *,std::allocator< amici::ExpData * > >": "ExpDataPtrVector",
    "std::vector< std::unique_ptr< amici::ReturnData >,std::allocator< "
    "std::unique_ptr< amici::ReturnData > > >": "Iterable[ReturnData]",
    "std::unique_ptr< amici::ExpData >": "ExpData",
    "std::unique_ptr< amici::ReturnData >": "ReturnData",
    "std::unique_ptr< amici::Solver >": "Solver",
    "amici::realtype": "float",
}

vector_types = {
    "IntVector": ":class:`int`",
    "BoolVector": ":class:`bool`",
    "DoubleVector": ":class:`float`",
    "StringVector": ":class:`str`",
    "ExpDataPtrVector": ":class:`amici.amici.ExpData`",
}


def process_docstring(app, what, name, obj, options, lines):
    # only apply in the amici.amici module
    if len(name.split(".")) < 2 or name.split(".")[1] != "amici":
        return

    # add custom doc to swig generated classes
    if len(name.split(".")) == 3 and name.split(".")[2] in [
        "IntVector",
        "BoolVector",
        "DoubleVector",
        "StringVector",
        "ExpDataPtrVector",
    ]:
        cname = name.split(".")[2]
        lines.append(
            f"Swig-Generated class templating common python "
            f"types including :class:`Iterable` "
            f"[{vector_types[cname]}] "
            f"and "
            f":class:`numpy.array` [{vector_types[cname]}] to facilitate"
            " interfacing with C++ bindings."
        )
        return

    if len(name.split(".")) == 3 and name.split(".")[2] in [
        "ExpDataPtr",
        "ReturnDataPtr",
        "ModelPtr",
        "SolverPtr",
    ]:
        cname = name.split(".")[2]
        lines.append(
            f"Swig-Generated class that implements smart pointers to "
            f'{cname.replace("Ptr", "")} as objects.'
        )
        return

    # add linebreaks before argument/return definitions
    lines_clean = []

    while len(lines):
        line = lines.pop(0)

        if (
            re.match(r":(type|rtype|param|return)", line)
            and len(lines_clean)
            and lines_clean[-1] != ""
        ):
            lines_clean.append("")

        lines_clean.append(line)
    lines.extend(lines_clean)

    for i in range(len(lines)):
        # fix types
        for old, new in typemaps.items():
            lines[i] = lines[i].replace(old, new)
        lines[i] = re.sub(
            r"amici::(Model|Solver|ExpData) ",
            r":class:`amici\.amici\.\1\`",
            lines[i],
        )
        lines[i] = re.sub(
            r"amici::(runAmiciSimulation[s]?)",
            r":func:`amici\.amici\.\1`",
            lines[i],
        )


def fix_typehints(sig: str) -> str:
    # cleanup types
    if not isinstance(sig, str):
        return sig

    for old, new in typemaps.items():
        sig = sig.replace(old, new)
    sig = sig.replace("void", "None")
    sig = sig.replace("amici::realtype", "float")
    sig = sig.replace("std::string", "str")
    sig = sig.replace("double", "float")
    sig = sig.replace("long", "int")
    sig = sig.replace("char const *", "str")
    sig = sig.replace("amici::", "")
    sig = sig.replace("sunindextype", "int")
    sig = sig.replace("H5::H5File", "object")

    # remove const
    sig = sig.replace(" const ", r" ")
    sig = re.sub(r" const$", r"", sig)

    # remove pass by reference
    sig = re.sub(r" &(,|\))", r"\1", sig)
    sig = re.sub(r" &$", r"", sig)

    # turn gsl_spans and pointers int Iterables
    sig = re.sub(r"([\w.]+) \*", r"Iterable[\1]", sig)
    sig = re.sub(r"gsl::span< ([\w.]+) >", r"Iterable[\1]", sig)

    # fix garbled output
    sig = sig.replace(" >", "")
    return sig


def process_signature(
    app, what: str, name: str, obj, options, signature, return_annotation
):
    if signature is None:
        return

    # only apply in the amici.amici module
    split_name = name.split(".")
    if len(split_name) < 2 or split_name[1] != "amici":
        return

    signature = fix_typehints(signature)
    if hasattr(obj, "__annotations__"):
        for ann in obj.__annotations__:
            obj.__annotations__[ann] = fix_typehints(obj.__annotations__[ann])

    return signature, return_annotation


# this code fixes references in symlinked md files in documentation folder
# link replacements must be in env.domains['std'].labels
doclinks = {
    "documentation/development": "/development.md",
    "documentation/CI": "/ci.md",
    "documentation/code_review_guide": "/code_review_guide.md",
}


def process_missing_ref(app, env, node, contnode):
    if not any(link in node["reftarget"] for link in doclinks):
        return  # speedup futile processing

    for old, new in doclinks.items():
        node["reftarget"] = node["reftarget"].replace(old, new)
    cnode = node[0]
    if "refuri" in cnode:
        for old, new in doclinks.items():
            cnode["refuri"] = cnode["refuri"].replace(old, new)

    refdoc = node.get("refdoc", env.docname)
    resolver = ReferencesResolver(env.get_doctree(refdoc))
    result = resolver.resolve_anyref(refdoc, node, cnode)
    return result


def skip_member(app, what, name, obj, skip, options):
    ignored_names = {
        "AbstractModel",
        "CVodeSolver",
        "IDASolver",
        "Model_ODE",
        "Model_DAE",
        "ConditionContext",
        "checkSigmaPositivity",
        "createGroup",
        "equals",
        "printErrMsgIdAndTxt",
        "wrapErrHandlerFn",
        "printWarnMsgIdAndTxt",
        "AmiciApplication",
        "writeReturnData",
        "writeReturnDataDiagnosis",
        "attributeExists",
        "locationExists",
        "createAndWriteDouble1DDataset",
        "createAndWriteDouble2DDataset",
        "createAndWriteDouble3DDataset",
        "createAndWriteInt1DDataset",
        "createAndWriteInt2DDataset",
        "createAndWriteInt3DDataset",
        "getDoubleDataset1D",
        "getDoubleDataset2D",
        "getDoubleDataset3D",
        "getIntDataset1D",
        "getIntScalarAttribute",
        "getDoubleScalarAttribute",
        "stdVec2ndarray",
        "SwigPyIterator",
        "thisown",
    }

    if name in ignored_names:
        return True

    if name.startswith("_") and name != "__init__":
        return True

    obj_str = str(obj)

    # ignore various functions for std::vector<> types
    if re.match(r"^<function [\w]+Vector\.", obj_str):
        return True

    # ignore various functions for smart pointer types
    if re.match(r"^<function [\w]+Ptr\.", obj_str):
        return True

    # ignore various functions for StringDoubleMap
    if obj_str.startswith("<function StringDoubleMap"):
        return True

    # Skip inherited members from builtins
    #  (skips, for example, all the int/str-derived methods of enums
    if (
        objclass := getattr(obj, "__objclass__", None)
    ) and objclass.__module__ == "builtins":
        return True

    # Avoid the following issue for all enum types:
    # > python/sdist/amici/amici.py:docstring of amici.amici.FixedParameterContext.from_bytes:9:
    #   WARNING: Inline interpreted text or phrase reference start-string without end-string.
    if (
        (qualname := getattr(obj, "__qualname__", ""))
        and qualname == "int.to_bytes"
    ) or (
        isinstance(getattr(obj, "__self__", None), EnumType)
        and name == "from_bytes"
    ):
        return True

    return None


def setup(app: "sphinx.application.Sphinx"):
    app.connect("autodoc-process-docstring", process_docstring, priority=0)
    app.connect("autodoc-process-signature", process_signature, priority=0)
    app.connect("missing-reference", process_missing_ref, priority=0)
    app.connect("autodoc-skip-member", skip_member, priority=0)
    app.config.intersphinx_mapping = intersphinx_mapping
    app.config.autosummary_generate = True
    app.config.autodoc_mock_imports = autodoc_mock_imports
