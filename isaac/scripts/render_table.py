"""
render_table.py
─────────────────────────────────────────────────────────────────────────────
Renders a standalone LaTeX table snippet (e.g. output of table.py) to PNG,
so you can eyeball it without building the whole thesis.

Usage
─────
    python render_table.py path/to/table.tex
    python render_table.py path/to/table.tex --out preview.png --dpi 200

Requires pdflatex and pdftoppm (poppler-utils) on PATH.
"""

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PREAMBLE = r"""
\documentclass{article}
\usepackage[margin=1in,landscape]{geometry}
\usepackage{booktabs}
\usepackage{amsmath}
\usepackage{amsbsy}
\usepackage{makecell}
\usepackage{pdflscape}
\usepackage[table]{xcolor}
\pagestyle{empty}
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("tex_path", type=str, help="Path to a .tex file containing one table "
                                                     "(e.g. \\begin{table}...\\end{table})")
    parser.add_argument("--out", type=str, default=None,
                         help="Output image path (default: <tex_path stem>.png next to the input)")
    parser.add_argument("--dpi", type=int, default=150)
    parser.add_argument("--preamble", type=str, default=None,
                         help="Path to a .tex file with extra \\usepackage/\\newcommand lines "
                              "(e.g. needed if the table uses custom macros like \\best)")
    args = parser.parse_args()

    tex_path = Path(args.tex_path).resolve()
    if not tex_path.exists():
        print(f"[ERROR] not found: {tex_path}")
        sys.exit(1)

    out_path = Path(args.out).resolve() if args.out else tex_path.with_suffix(".png")

    extra_preamble = Path(args.preamble).read_text() if args.preamble else ""

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        shutil.copy(tex_path, tmp / "table_content.tex")

        (tmp / "preview.tex").write_text(
            PREAMBLE + extra_preamble +
            "\n\\begin{document}\n\\input{table_content}\n\\end{document}\n"
        )

        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "preview.tex"],
            cwd=tmp, capture_output=True, text=True,
        )
        if result.returncode != 0 or not (tmp / "preview.pdf").exists():
            print("[ERROR] pdflatex failed:")
            print("\n".join(result.stdout.splitlines()[-40:]))
            sys.exit(1)

        subprocess.run(
            ["pdftoppm", "-png", "-r", str(args.dpi), "preview.pdf", "preview"],
            cwd=tmp, check=True,
        )

        pages = sorted(tmp.glob("preview-*.png"))
        if not pages:
            print("[ERROR] pdftoppm produced no output")
            sys.exit(1)

        if len(pages) == 1:
            shutil.copy(pages[0], out_path)
            print(f"[OK] saved -> {out_path}")
        else:
            for i, p in enumerate(pages, start=1):
                dest = out_path.with_name(f"{out_path.stem}-{i}{out_path.suffix}")
                shutil.copy(p, dest)
                print(f"[OK] saved -> {dest}")


if __name__ == "__main__":
    main()
