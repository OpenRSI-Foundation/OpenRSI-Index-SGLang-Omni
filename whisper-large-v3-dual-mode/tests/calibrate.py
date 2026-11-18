"""Optional GPU recalibration helper; emits RSI_BASELINE_JSON=JSON without writing reward.json."""
import sys
sys.path.insert(0, '/tests')
from evaluate import run_cli
run_cli(calibrate=True)
