#!/usr/bin/env python3
"""
Create AMICI publication list by year

For macOS with clang: download and install biblib-simple to avoid compiler
issues

Requires pandoc
"""

import os
import subprocess
import sys

import biblib.algo
import biblib.bib
import biblib.messages


def get_keys_by_year(bibfile):
    """Get bibtex entry keys as dict by year"""

    with open(bibfile) as f:
        db = biblib.bib.Parser().parse(f, log_fp=sys.stderr).get_entries()
        recoverer = biblib.messages.InputErrorRecoverer()
        by_year = {}
        for ent in db.values():
            with recoverer:
                if "year" in ent:
                    try:
                        by_year[ent["year"]].append(ent.key)
                    except KeyError:
                        by_year[ent["year"]] = [ent.key]
                else:
                    print("Missing year for entry", ent.key)
        recoverer.reraise()
    return by_year


def get_sub_bibliography(year, by_year, bibfile):
    """Get HTML bibliography for the given year"""

    entries = ",".join(["@" + x for x in by_year[year]])
    stdin_input = (
        "---\n"
        f"bibliography: {bibfile}\n"
        f'nocite: "{entries}"\n...\n'
        f"# {year}"
    )

    out = subprocess.run(
        ["pandoc", "--citeproc", "-f", "markdown"],
        input=stdin_input,
        capture_output=True,
        encoding="utf-8",
    )
    if out.returncode != 0:
        raise AssertionError(out.stderr)

    return out.stdout


def main():
    script_path = os.path.dirname(os.path.realpath(__file__))
    bibfile = os.path.join(script_path, "amici_refs.bib")
    outfile = os.path.join(script_path, "references.md")

    by_year = get_keys_by_year(bibfile)
    num_total = sum(map(len, by_year.values()))
    with open(outfile, "w") as f:
        f.write("# References\n\n")
        f.write(
            "List of publications using AMICI. "
            f"Total number is {num_total}.\n\n"
        )
        f.write(
            "If you applied AMICI in your work and your publication is "
            "missing, please let us know via a new\n"
            "[GitHub issue](https://github.com/AMICI-dev/AMICI/issues/new"
            "?labels=documentation&title=Add+publication"
            "&body=AMICI+was+used+in+this+manuscript:+DOI).\n\n"
        )
        f.write(
            """
<style>
.csl-entry {
    padding: 5px
}
</style>\n
"""
        )
        for year in reversed(sorted(by_year.keys())):
            cur_bib = get_sub_bibliography(year, by_year, bibfile)
            f.write(cur_bib)


if __name__ == "__main__":
    main()
