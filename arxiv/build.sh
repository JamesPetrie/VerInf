#!/bin/bash
# Build the arXiv-preprint-style PDF from ../paper.md and install it as the
# repo-root paper.pdf (the file the paper.md banner links to).
set -e
cd "$(dirname "$0")"
pandoc -f markdown-markdown_in_html_blocks ../paper.md --highlight-style=tango -o paper-body.tex
python3 convert.py
pdflatex -interaction=nonstopmode paper-arxiv.tex > build.log 2>&1 || { grep -m5 "^!" build.log; exit 1; }
pdflatex -interaction=nonstopmode paper-arxiv.tex >> build.log 2>&1
cp paper-arxiv.pdf ../paper.pdf
echo "OK: $(pdfinfo ../paper.pdf 2>/dev/null | grep Pages || echo built)"
