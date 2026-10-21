#!/usr/bin/env bash
# CPU experiments; the RSNA post hoc results need the out-of-fold scores (see README).
set -e
cd "$(dirname "$0")"
python experiments/e1_simulation.py
python experiments/make_figure.py > /dev/null
python experiments/e3_published.py
python experiments/rsna_prespecified.py
python experiments/cbis_case.py
python audit.py
