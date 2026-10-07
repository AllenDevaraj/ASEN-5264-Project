#!/usr/bin/env python3
"""Run queued jobs one at a time under the watchdog; resumable.

queue.json: [{"name": "oracle_s0", "cmd": ["./vpy", "-m", "pomdp_pick.agents.train", ...]}, ...]
Finished names are appended to queue.json.done; rerunning skips them.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from watchdog import run  # noqa: E402


def main(queue_path):
    jobs = json.load(open(queue_path))
    done_path = queue_path + ".done"
    done = set(open(done_path).read().split()) if os.path.exists(done_path) else set()
    for job in jobs:
        if job["name"] in done:
            continue
        rc = run(job["cmd"], log_path=os.path.join("runs", "watchdog.log"))
        if rc != 0:
            print(f"job {job['name']} failed with {rc}; stopping queue", file=sys.stderr)
            sys.exit(rc)
        with open(done_path, "a") as fh:
            fh.write(job["name"] + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
