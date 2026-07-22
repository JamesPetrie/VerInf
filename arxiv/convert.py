#!/usr/bin/env python3
"""Assemble the arXiv-preprint-style LaTeX source for VerInf.

paper.md stays the source of truth. pandoc converts it to a LaTeX body;
this script extracts the title (the markdown H1), drops the byline and
horizontal rule (the \\maketitle block replaces them), shifts heading
levels up one (the markdown uses ## for sections), pulls a provisional
abstract (the introduction's first paragraph, duplicated), strips the
"Figure N:" caption prefixes (LaTeX numbers captions itself), and wraps
everything in preamble.tex (PRIMEarxiv style, same as off-switch).

Build: ./build.sh
"""
import os
import re

with open("paper-body.tex") as f:
    body = f.read()

# Strip \hypertarget wrappers (older pandoc) and heading labels.
body = re.sub(
    r"\\hypertarget\{[^}]+\}\{%\s*\n(\\(?:section|subsection|subsubsection|paragraph)\*?\{(?:[^{}]|\{[^{}]*\})*?\})\\label\{[^}]+\}\}",
    r"\1",
    body,
)
body = re.sub(r"\\label\{[^}]+\}\}", "}", body)

# Title: the markdown H1 became the body's first \section. Pull it out.
title_match = re.search(r"\\section\{((?:[^{}]|\{[^{}]*\})*)\}\\label\{[^}]*\}", body)
if not title_match:
    title_match = re.search(r"\\section\{((?:[^{}]|\{[^{}]*\})*)\}", body)
title = title_match.group(1).strip()
body = body[: title_match.start()] + body[title_match.end():]

# Drop the author byline paragraph and the horizontal rule after the title.
body = re.sub(r"^\s*James Petrie\s*$", "", body, count=1, flags=re.MULTILINE)
body = re.sub(
    r"\\begin\{center\}\\rule\{0\.5\\linewidth\}\{0\.5pt\}\\end\{center\}",
    "",
    body,
    count=1,
)

# Shift heading levels up one: markdown ## sections arrive as \subsection.
body = re.sub(r"\\paragraph\{", r"\\subsubsection{", body)
body = re.sub(r"\\subsubsection\{", r"__TMP_SUBSEC__{", body)
body = re.sub(r"\\subsection\{", r"\\section{", body)
body = re.sub(r"__TMP_SUBSEC__\{", r"\\subsection{", body)

# Provisional abstract: the introduction's first paragraph, duplicated.
intro_match = re.search(
    r"\\section\{1\. Introduction\}\s*\n\n?(.*?)(?=\n\n)", body, re.DOTALL
)
abstract = intro_match.group(1).strip() if intro_match else "TODO: abstract"

# Figures: strip pandoc's height cap and the "Figure N:" caption prefix,
# and point paths at the repo's analysis/figures/.
body = re.sub(r"\\caption\{Figure\s+\d+[:.]\s*", r"\\caption{", body)
body = re.sub(r",height=\\textheight", "", body)
body = body.replace("{analysis/figures/", "{../analysis/figures/")

# Strip stray \texorpdfstring (keep the TeX arm).
body = re.sub(r"\\texorpdfstring\{([^{}]*)\}\{[^{}]*\}", r"\1", body)

# Appendices start on a fresh page.
body = re.sub(r"(\\section\{Appendix)", r"\\clearpage\n\1", body, count=1)

script_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(script_dir, "preamble.tex")) as f:
    preamble = f.read()

out = preamble.replace("__TITLE__", title).replace("__ABSTRACT__", abstract)
out += body
out += "\n\\end{document}\n"

with open(os.path.join(script_dir, "paper-arxiv.tex"), "w") as f:
    f.write(out)

print(f"Title: {title}")
print(f"Abstract length: {len(abstract)} chars")
print(f"Body length: {len(body)} chars")
