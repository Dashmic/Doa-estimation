import os, sys, io, runpy

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
demo = os.path.join(HERE, "demo_interactive.py")
checkpoint = os.path.join(HERE, "checkpoints", "cov_t32_best.pth")

if not os.path.isfile(demo):
    sys.exit(f"ERROR: demo_interactive.py not found at: {demo!r}")

if not os.path.isfile(checkpoint):
    sys.exit(f"ERROR: Checkpoint not found at: {checkpoint!r}")

user_args = sys.argv[1:]

if "--checkpoint" not in user_args:
    user_args = ["--checkpoint", checkpoint] + user_args

if "--demo-mode" not in user_args and "--angles" not in user_args:
    user_args = ["--demo-mode"] + user_args

sys.argv = [demo] + user_args
runpy.run_path(demo, run_name="__main__")
