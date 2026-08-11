#!/bin/bash
# Build paper.docx from paper.md in the gdoc house style.
# The Lua filter rewrites the Appendix B claim-listing arrays as real Word
# tables and splits aligned blocks into per-line equations, because Google
# Docs' equation importer has no matrix support and drops OMML matrices.
set -e
cd "$(dirname "$0")/.."
pandoc -f markdown-markdown_in_html_blocks paper.md \
  --reference-doc=gdoc-reference.docx --highlight-style=tango \
  --lua-filter=tools/arrays-to-tables.lua -o paper.docx
echo "OK: paper.docx"
