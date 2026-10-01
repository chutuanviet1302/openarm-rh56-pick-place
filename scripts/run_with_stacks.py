"""Run a module with a periodic stack dump (faulthandler) -- to see where a long run spends its time.

    python -m scripts.run_with_stacks --every 120 --stacks artifacts/runs/r2_stacks.txt -- simulation.pick_place.bin_conveyor_task --no-view
"""

import argparse
import faulthandler
import runpy
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--every", type=float, default=120.0)
parser.add_argument("--stacks", required=True)
parser.add_argument("module")
parser.add_argument("args", nargs=argparse.REMAINDER)
args = parser.parse_args()
handle = open(args.stacks, "w")
faulthandler.dump_traceback_later(args.every, repeat=True, file=handle)
sys.argv = [args.module, *args.args]
runpy.run_module(args.module, run_name="__main__", alter_sys=True)
