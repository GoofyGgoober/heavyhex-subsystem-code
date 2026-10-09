#!/bin/sh
# Builds the paper and packs what arXiv needs into arxiv.tar.gz. arXiv does not
# run BibTeX, so the bundle carries main.bbl instead of refs.bib.
set -e
cd "$(dirname "$0")"
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
tar czf arxiv.tar.gz main.tex main.bbl figures/*.pdf
