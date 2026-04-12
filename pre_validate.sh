#!/usr/bin/env bash
# =============================================================================
# pre_validate.sh — InventOps Pre-Submission Validator
#
# Runs all checks locally before running the official validate-submission.sh
#
# Usage:
#   chmod +x pre_validate.sh
#   ./pre_validate.sh <hf_space_url> [repo_dir]
#
# Examples:
#   ./pre_validate.sh https://myuser-inventops.hf.space
#   ./pre_validate.sh https://myuser-inventops.hf.space ./inventops
# =============================================================================

set -uo pipefail

# ── Colors ────────────────────────────────────────────────────────────────────
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
  CYAN='\033[0;36m'; BOLD='\033[1m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; CYAN=''; BOLD=''; NC=''
fi

# ── Args ──────────────────────────────────────────────────────────────────────
HF_URL="${1:-}"
REPO_DIR="${2:-.}"

if [ -z "$HF_URL" ]; then
  printf "${BOLD}Usage:${NC} %s <hf_space_url> [repo_dir]\n" "$0"
  printf "\n  Example: %s https://myuser-inventops.hf.space ./inventops\n\n" "$0"
  exit 1
fi

HF_URL="${HF_URL%/}"   # strip trailing slash

if ! REPO_DIR="$(cd "$REPO_DIR" 2>/dev/null && pwd)"; then
  printf "${RED}Error:${NC} directory '%s' not found\n" "${2:-.}"
  exit 1
fi

# ── Helpers ───────────────────────────────────────────────────────────────────
PASS=0; FAIL=0

log()  { printf "[%s] %b\n"      "$(date -u +%H:%M:%S)" "$*"; }
pass() { log "${GREEN}✓ PASS${NC}  $1"; PASS=$((PASS+1)); }
fail() { log "${RED}✗ FAIL${NC}  $1"; FAIL=$((FAIL+1)); }
warn() { log "${YELLOW}⚠ WARN${NC}  $1"; }
info() { log "${CYAN}ℹ INFO${NC}  $1"; }
section() {
  printf "\n${BOLD}━━━ %s ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n" "$1"
}

# ── Banner ────────────────────────────────────────────────────────────────────
printf "\n${BOLD}╔══════════════════════════════════════════════════════╗${NC}\n"
printf "${BOLD}║      InventOps — Pre-Submission Validator            ║${NC}\n"
printf "${BOLD}╚══════════════════════════════════════════════════════╝${NC}\n"
info "Repo dir : $REPO_DIR"
info "HF URL   : $HF_URL"

# =============================================================================
# CHECK 1 — Required files exist
# =============================================================================
section "1/6  Required files"

REQUIRED_FILES=(
  "inference.py"
  "openenv.yaml"
  "Dockerfile"
  "requirements.txt"
  "server.py"
  "InventOps/__init__.py"
  "InventOps/env.py"
  "InventOps/models.py"
  "InventOps/simulator.py"
  "InventOps/reward.py"
  "InventOps/demand.py"
  "InventOps/data/demand_profiles.json"
  "InventOps/tasks/base.py"
  "InventOps/tasks/task_easy.py"
  "InventOps/tasks/task_medium.py"
  "InventOps/tasks/task_hard.py"
  "rlvr/agent.py"
  "rlvr/prompts/base_prompt.txt"
)

ALL_PRESENT=true
for f in "${REQUIRED_FILES[@]}"; do
  if [ -f "$REPO_DIR/$f" ]; then
    pass "Found: $f"
  else
    fail "Missing: $f"
    ALL_PRESENT=false
  fi
done

if [ "$ALL_PRESENT" = false ]; then
  printf "\n${RED}One or more required files are missing. Fix before continuing.${NC}\n\n"
  exit 1
fi

# =============================================================================
# CHECK 2 — Python environment & imports
# =============================================================================
section "2/6  Python imports"

if ! command -v python3 &>/dev/null; then
  fail "python3 not found — install Python 3.11+"
  exit 1
else
  PY_VERSION=$(python3 --version 2>&1)
  pass "Python: $PY_VERSION"
fi

# Install deps quietly
info "Installing requirements..."
if pip install -r "$REPO_DIR/requirements.txt" -q --break-system-packages 2>/dev/null \
   || pip install -r "$REPO_DIR/requirements.txt" -q 2>/dev/null; then
  pass "pip install requirements.txt"
else
  fail "pip install failed"
fi

# Test core imports
cd "$REPO_DIR"
IMPORT_TEST=$(PYTHONPATH="$REPO_DIR" python3 -c "
from InventOps import SupplyChainEnv
from InventOps.models import Action, Observation
from InventOps.reward import compute_reward
from InventOps.demand import DemandGenerator
print('OK')
" 2>&1)

if [ "$IMPORT_TEST" = "OK" ]; then
  pass "All core imports succeed"
else
  fail "Import error: $IMPORT_TEST"
fi

# =============================================================================
# CHECK 3 — Environment smoke test (reset / step / grade)
# =============================================================================
section "3/6  Environment smoke test"

ENV_TEST=$(PYTHONPATH="$REPO_DIR" python3 - << 'PYEOF'
import sys
from InventOps import SupplyChainEnv
from InventOps.models import Action, Observation

errors = []
for task_id in ["easy", "medium", "hard"]:
    try:
        env = SupplyChainEnv(task_id=task_id, seed=42)
        obs = env.reset()

        # reset() returns Observation
        assert isinstance(obs, Observation), f"{task_id}: reset() must return Observation"
        assert obs.day == 0, f"{task_id}: day must be 0 after reset"

        # step() contract
        obs2, reward, done, info = env.step(Action(action_type="hold"))
        assert isinstance(obs2, Observation), f"{task_id}: step() obs must be Observation"
        assert isinstance(reward, float),     f"{task_id}: step() reward must be float"
        assert isinstance(done, bool),        f"{task_id}: step() done must be bool"
        assert isinstance(info, dict),        f"{task_id}: step() info must be dict"

        # Run to completion
        while not done:
            obs2, reward, done, info = env.step(Action(action_type="hold"))

        # grade() contract
        score = env.grade()
        assert isinstance(score, float), f"{task_id}: grade() must return float"
        assert 0.0 <= score <= 1.0,      f"{task_id}: grade() must be in [0,1], got {score}"

        # Determinism: same seed → same score
        env2 = SupplyChainEnv(task_id=task_id, seed=42)
        env2.reset()
        while True:
            _, _, done2, _ = env2.step(Action(action_type="hold"))
            if done2:
                break
        score2 = env2.grade()
        assert score == score2, f"{task_id}: grader not deterministic ({score} vs {score2})"

        print(f"  {task_id}: score={score:.4f} PASS")

    except Exception as e:
        errors.append(f"{task_id}: {e}")
        print(f"  {task_id}: FAIL — {e}")

if errors:
    sys.exit(1)
PYEOF
)
EXIT_CODE=$?

echo "$ENV_TEST"
if [ $EXIT_CODE -eq 0 ]; then
  pass "All 3 tasks: reset/step/grade contracts verified, scores in [0,1], deterministic"
else
  fail "Environment smoke test failed — fix before submitting"
fi

# =============================================================================
# CHECK 4 — inference.py stdout format
# =============================================================================
section "4/6  inference.py stdout format"

INFER_TEST=$(PYTHONPATH="$REPO_DIR" python3 - << 'PYEOF'
import sys, json, unittest.mock as mock, os

os.environ.setdefault("HF_TOKEN", "dummy")

# Mock OpenAI to return hold actions without a real API call
mock_completion = mock.MagicMock()
mock_completion.choices[0].message.content = '{"action_type": "hold"}'

with mock.patch("openai.OpenAI") as MockOpenAI:
    client_instance = MockOpenAI.return_value
    client_instance.chat.completions.create.return_value = mock_completion

    # Redirect stdout to capture
    from io import StringIO
    captured = StringIO()
    real_stdout = sys.stdout
    sys.stdout = captured

    # Remove cached module if any
    for key in list(sys.modules.keys()):
        if "inference" in key:
            del sys.modules[key]

    import inference
    inference.run_episode(client_instance, "easy")

    sys.stdout = real_stdout
    output = captured.getvalue()

lines = [l for l in output.strip().splitlines() if l.startswith("[")]
errors = []

start_lines  = [l for l in lines if l.startswith("[START]")]
step_lines   = [l for l in lines if l.startswith("[STEP]")]
end_lines    = [l for l in lines if l.startswith("[END]")]

if len(start_lines) != 1:
    errors.append(f"Expected 1 [START] line, got {len(start_lines)}")
if len(end_lines) != 1:
    errors.append(f"Expected 1 [END] line, got {len(end_lines)}")
if len(step_lines) != 30:
    errors.append(f"Expected 30 [STEP] lines for easy task, got {len(step_lines)}")

# Validate [START] fields
start = start_lines[0] if start_lines else ""
for field in ["task=", "env=", "model="]:
    if field not in start:
        errors.append(f"[START] missing field: {field}")

# Validate a [STEP] line
if step_lines:
    step = step_lines[0]
    for field in ["step=", "action=", "reward=", "done=", "error="]:
        if field not in step:
            errors.append(f"[STEP] missing field: {field}")
    # done must be lowercase
    if "done=True" in step or "done=False" in step:
        errors.append("[STEP] done= must be lowercase true/false")

# Validate [END] fields
end = end_lines[0] if end_lines else ""
for field in ["success=", "steps=", "score=", "rewards="]:
    if field not in end:
        errors.append(f"[END] missing field: {field}")
if "success=True" in end or "success=False" in end:
    errors.append("[END] success= must be lowercase true/false")

if errors:
    for e in errors:
        print(f"  FAIL: {e}")
    sys.exit(1)
else:
    print(f"  [START] line  : OK")
    print(f"  [STEP]  lines : {len(step_lines)} (expected 30)")
    print(f"  [END]   line  : OK")
    print(f"  Field names   : OK")
    print(f"  Bool format   : OK (lowercase)")
PYEOF
)
EXIT_CODE=$?

echo "$INFER_TEST"
if [ $EXIT_CODE -eq 0 ]; then
  pass "inference.py stdout format is spec-compliant"
else
  fail "inference.py stdout format has errors"
fi

# =============================================================================
# CHECK 5 — Docker build
# =============================================================================
section "5/6  Docker build"

if ! command -v docker &>/dev/null; then
  warn "Docker not found — skipping build check"
  warn "Install Docker: https://docs.docker.com/get-docker/"
else
  info "Running docker build (this may take a minute)..."
  if docker build -t inventops-validate "$REPO_DIR" > /tmp/docker_build.log 2>&1; then
    pass "docker build succeeded"
    # Clean up test image
    docker rmi inventops-validate -f > /dev/null 2>&1 || true
  else
    fail "docker build failed — last 20 lines:"
    tail -20 /tmp/docker_build.log | sed 's/^/  /'
  fi
fi

# =============================================================================
# CHECK 6 — HF Space ping
# =============================================================================
section "6/6  HF Space ping (POST /reset)"

if ! command -v curl &>/dev/null; then
  warn "curl not found — skipping Space ping"
else
  info "Pinging $HF_URL/reset ..."
  HTTP_CODE=$(curl -s -o /tmp/hf_ping.out -w "%{http_code}" \
    -X POST "$HF_URL/reset" \
    -H "Content-Type: application/json" \
    -d '{}' \
    --max-time 30 2>/dev/null || echo "000")

  if [ "$HTTP_CODE" = "200" ]; then
    BODY=$(cat /tmp/hf_ping.out 2>/dev/null || echo "")
    pass "HF Space responded 200 to POST /reset"
    info "Response: $BODY"
  elif [ "$HTTP_CODE" = "000" ]; then
    fail "Could not reach $HF_URL — connection failed or timed out"
    warn "Make sure your HF Space is deployed and running"
    warn "Check Space logs at: https://huggingface.co/spaces"
  else
    fail "POST /reset returned HTTP $HTTP_CODE (expected 200)"
    warn "Space may still be booting — wait 2-3 min and retry"
    warn "Check Space logs at: https://huggingface.co/spaces"
  fi
fi

# =============================================================================
# Summary
# =============================================================================
printf "\n${BOLD}━━━ Summary ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
TOTAL=$((PASS + FAIL))
printf "  Passed : ${GREEN}${BOLD}%d${NC}\n" "$PASS"
printf "  Failed : ${RED}${BOLD}%d${NC}\n" "$FAIL"
printf "  Total  : %d checks\n\n" "$TOTAL"

if [ "$FAIL" -eq 0 ]; then
  printf "${GREEN}${BOLD}  ✓ All checks passed — ready to run validate-submission.sh${NC}\n"
  printf "\n  Next step:\n"
  printf "  ${CYAN}./validate-submission.sh %s %s${NC}\n\n" "$HF_URL" "$REPO_DIR"
  exit 0
else
  printf "${RED}${BOLD}  ✗ %d check(s) failed — fix before submitting${NC}\n\n" "$FAIL"
  exit 1
fi
