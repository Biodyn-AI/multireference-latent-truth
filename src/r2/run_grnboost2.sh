#!/bin/zsh
# Full, re-runnable sequence for stage grnboost2 (outputs/r2/grnboost2/), about 60 min.
# time -l: user/sys CPU include the dask workers; max RSS = largest single process.
# Paths are resolved relative to this script; override the interpreters with
#   PY=<python with arboreto 0.1.6, dask, scikit-learn 1.5.1>
#   R2_SKLEARN_REF_PYTHON=<python with scikit-learn <= 1.2 for validate-es>
set -e
S=${0:A:h:h:h}                       # subproject root (src/r2/ -> ../..)
BW=${S:h}                            # biodyn-work root
PY=${PY:-$BW/.venv/bin/python}
REF=${R2_SKLEARN_REF_PYTHON:-$HOME/anaconda3/envs/econ/bin/python}
B=$S/src/r2/build_grnboost2.py
O=$S/outputs/r2/grnboost2
mkdir -p $O/logs
for t in immune kidney lung; do
  mkdir -p $O/$t
  /usr/bin/time -l -o $O/$t/time_l_grnboost2.txt $PY $B run --tissue $t --method grnboost2 > $O/logs/grnboost2_$t.log 2>&1
  echo "done grnboost2 $t $(date)"
  /usr/bin/time -l -o $O/$t/time_l_grnboost2_detect1pct.txt $PY $B run --tissue $t --method grnboost2 --min-detect-frac 0.01 > $O/logs/grnboost2_detect1pct_$t.log 2>&1
  echo "done grnboost2_detect1pct $t $(date)"
done
if [[ -x $REF ]]; then
  $PY $B validate-es --ref-python $REF > $O/logs/validate_es.log 2>&1
else
  $PY $B validate-es > $O/logs/validate_es.log 2>&1
fi
echo "done validate-es $(date)"
/usr/bin/time -l -o $O/immune/time_l_genie3.txt $PY $B run --tissue immune --method genie3 > $O/logs/genie3_immune.log 2>&1
echo "done genie3 immune $(date)"
# GENIE3 sensitivity (decision-log item 9): >=1%-detection gene filter
/usr/bin/time -l -o $O/immune/time_l_genie3_detect1pct.txt $PY $B run --tissue immune --method genie3 --min-detect-frac 0.01 > $O/logs/genie3_detect1pct_immune.log 2>&1
echo "done genie3_detect1pct immune $(date)"
$PY $B table > $O/logs/table.log 2>&1
$PY $B summary > $O/logs/summary.log 2>&1
echo "done table + summary $(date)"
