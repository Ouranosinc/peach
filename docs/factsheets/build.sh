#!/bin/bash -x
set -xe
shopt -s extglob

mkdir -p  /quarto-run/en
cd /quarto-run/en
find . -maxdepth 1 -type l -delete
ln -s /input/en/* .
ln -s /quarto-extensions/* .
quarto render --to html --output-dir /output/en

mkdir -p  /quarto-run/fr
cd /quarto-run/fr
find . -maxdepth 1 -type l -delete
ln -s /input/fr/* .
ln -s /quarto-extensions/* .
quarto render --to html --output-dir /output/fr

cd /quarto-run