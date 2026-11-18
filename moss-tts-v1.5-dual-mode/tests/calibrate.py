"""Optional GPU recalibration helper; emits RSI_BASELINE_JSON=JSON without writing reward.json."""
import sys
sys.path.insert(0, '/tests')
from engine import main
raise SystemExit(main(calibrate=True))
