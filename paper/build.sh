#!/usr/bin/env bash
# Build the manuscript PDF and the S1 Text PDF with pdflatex + bibtex + pdflatex x2.
#
#   ./build.sh             main.tex and s1_text.tex -> build/main.pdf, build/s1_text.pdf
#   ./build.sh --no-figs   main.tex without the draft \FIG images (PLOS: no graphics in the
#                          manuscript PDF) -> build/main.pdf; s1_text.pdf keeps its figures
#   ./build.sh main        only main.tex (or: s1_text)
#
# pdflatex runs in tex/ so that \input{sections/...} and ../figures/ resolve; all output goes
# to build/. bibtex runs in build/ (it refuses to write outside its working directory).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEX="$HERE/tex"
OUT="$HERE/build"
mkdir -p "$OUT"

NOFIGS=0
DOCS=()
for a in "$@"; do
  case "$a" in
    --no-figs) NOFIGS=1 ;;
    main|s1_text) DOCS+=("$a") ;;
    *) echo "usage: build.sh [--no-figs] [main] [s1_text]" >&2; exit 2 ;;
  esac
done
[ ${#DOCS[@]} -eq 0 ] && DOCS=(main s1_text)

# The two files must share one preamble (everything up to the macros section).
if ! diff -q <(sed -n '/^\\documentclass/,/^%% Include all user-defined macros below/p' "$TEX/main.tex") \
             <(sed -n '/^\\documentclass/,/^%% Include all user-defined macros below/p' "$TEX/s1_text.tex") >/dev/null; then
  echo "WARNING: the preambles of main.tex and s1_text.tex differ" >&2
fi

run_latex() {   # $1 = document stem, $2 = 1 to switch draft figures off
  # \pdfsuppressptexinfo keeps the absolute paths of included figures out of the PDF.
  local stem="$1" arg='\pdfsuppressptexinfo=-1 \input{'"$1"'}'
  [ "$2" = 1 ] && arg='\pdfsuppressptexinfo=-1 \AtBeginDocument{\draftfigsfalse}\input{'"$stem"'}'
  ( cd "$TEX" && pdflatex -interaction=nonstopmode -halt-on-error -jobname="$stem" \
      -output-directory="$OUT" "$arg" ) > "$OUT/$stem.pdflatex.out" 2>&1
}

STATUS=0
for stem in "${DOCS[@]}"; do
  nofigs=0
  [ "$stem" = main ] && nofigs=$NOFIGS
  rm -f "$OUT/$stem.aux" "$OUT/$stem.bbl" "$OUT/$stem.pdf"
  if ! run_latex "$stem" "$nofigs"; then
    echo "pdflatex failed on $stem.tex (pass 1):" >&2
    grep -A3 '^!' "$OUT/$stem.log" | head -30 >&2
    STATUS=1; continue
  fi
  if grep -q '^\\citation' "$OUT/$stem.aux"; then
    if ! ( cd "$OUT" && BIBINPUTS="$TEX:" BSTINPUTS="$TEX:" bibtex "$stem" ) > "$OUT/$stem.bibtex.out" 2>&1; then
      echo "bibtex reported errors for $stem (see build/$stem.blg):" >&2
      grep -B1 -A3 -i 'error\|^I ' "$OUT/$stem.blg" | head -30 >&2
      STATUS=1
    fi
    grep '^Warning--' "$OUT/$stem.blg" >&2 || true
  else
    echo "$stem: no citations yet, bibtex skipped"
    # an empty reference list: keeps \bibliography from failing on a missing .bbl
    printf '%s\n' '\begin{thebibliography}{1}' '\item[] ' '\end{thebibliography}' > "$OUT/$stem.bbl"
  fi
  ok=1
  for pass in 2 3; do
    if ! run_latex "$stem" "$nofigs"; then
      echo "pdflatex failed on $stem.tex (pass $pass):" >&2
      grep -A3 '^!' "$OUT/$stem.log" | head -30 >&2
      STATUS=1; ok=0; break
    fi
  done
  [ $ok = 1 ] || continue
  pages=$(pdfinfo "$OUT/$stem.pdf" 2>/dev/null | sed -n 's/^Pages: *//p')
  undef=$(grep -c 'Citation .* undefined\|Reference .* undefined' "$OUT/$stem.log" || true)
  gaps=$(pdftotext "$OUT/$stem.pdf" - 2>/dev/null | grep -o '\[GAP:' | wc -l)
  missing=$(pdftotext "$OUT/$stem.pdf" - 2>/dev/null | grep -c 'missing figure file' || true)
  echo "$stem: $OUT/$stem.pdf ($pages pages; undefined citations/references: $undef; GAP markers: $gaps; missing figure files: $missing)"
done
exit $STATUS
