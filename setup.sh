#!/bin/sh
export PYTHONPATH=$(dirname "$0")
source venv/bin/activate
pip install -r requirements.txt
python3 Visualization/GenericScreen/generic_screen_family_test.py