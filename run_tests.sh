#!/usr/bin/env bash
set -e
VENV=/home/aryanshr/projects/InventOps/.venv/bin/python
ROOT=/home/aryanshr/projects/InventOps

echo "=== pytest suite ==="
$VENV -m pytest $ROOT/tests/grader.py $ROOT/tests/env_interface.py $ROOT/tests/action_validation.py $ROOT/tests/demand.py $ROOT/tests/reward.py -v --tb=short

echo ""
echo "=== inference self-test ==="
$VENV $ROOT/tests/inference_test.py --test
