#!/usr/bin/env python3
"""Run a command under laptop-safety limits.

Pauses the whole process group (SIGSTOP) when the CPU package is hot, memory is low
or the laptop is on battery; resumes (SIGCONT) when safe again. The child runs at nice 19.
Usage: python tools/watchdog.py [--log runs/watchdog.log] -- cmd args...
"""
import argparse
import glob
import os
import signal
import subprocess
import sys
import time

TEMP_PAUSE_C = 85.0       # applied to the ~30 s average package temperature
TEMP_RESUME_C = 75.0
TEMP_HARD_C = 95.0        # instantaneous cap (TjMax is 100 °C): react immediately above this
MEM_PAUSE_MB = 2000.0
MEM_KILL_MB = 700.0
MEM_STALL_S = 600.0       # paused for low memory this long: the job itself holds the memory, stop it


def decide(paused, temp_c, mem_avail_mb, on_ac):
    """Pure policy: 'pause' | 'resume' | 'none'. Unknown readings (None) count as safe."""
    hot = temp_c is not None and temp_c > TEMP_PAUSE_C
    cool = temp_c is None or temp_c < TEMP_RESUME_C
    low_mem = mem_avail_mb is not None and mem_avail_mb < MEM_PAUSE_MB
    battery = on_ac is False
    if not paused:
        return "pause" if (hot or low_mem or battery) else "none"
    return "resume" if (cool and not low_mem and not battery) else "none"


class ThermalFilter:
    """Average recent package temperatures so turbo-boost spikes don't cause pause/resume
    thrash, while sustained heat (or anything above TEMP_HARD_C) still pauses at once."""

    def __init__(self, window=6):
        self.window, self.samples = window, []

    def update(self, temp_c):
        if temp_c is None:
            return None
        self.samples = (self.samples + [temp_c])[-self.window:]
        if temp_c >= TEMP_HARD_C:
            return temp_c
        return sum(self.samples) / len(self.samples)


def read_temp_c():
    for zone in glob.glob("/sys/class/thermal/thermal_zone*"):
        try:
            if open(f"{zone}/type").read().strip() == "x86_pkg_temp":
                return int(open(f"{zone}/temp").read()) / 1000.0
        except OSError:
            continue
    return None


def read_mem_avail_mb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1024.0
    except OSError:
        pass
    return None


def read_on_ac():
    for path in glob.glob("/sys/class/power_supply/*/online"):
        if "AC" in path or "ADP" in path:
            try:
                return open(path).read().strip() == "1"
            except OSError:
                return None
    return None


def run(cmd, log_path, poll_s=5.0, readers=None, mem_stall_s=MEM_STALL_S):
    log_path = str(log_path)
    os.makedirs(os.path.dirname(log_path) or ".", exist_ok=True)
    log = open(log_path, "a", buffering=1)
    stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S")  # noqa: E731
    log.write(f"{stamp()} start {' '.join(cmd)}\n")
    proc = subprocess.Popen(cmd, start_new_session=True, preexec_fn=lambda: os.nice(19))
    paused = False
    stop = []
    thermal = ThermalFilter(window=max(1, int(30 / poll_s)))
    read_t, read_m, read_ac = readers or (read_temp_c, read_mem_avail_mb, read_on_ac)
    low_mem_since = None

    def _on_signal(signum, _frame):
        stop.append(signum)

    # If the watchdog is told to stop, never leave its (possibly paused) job frozen behind.
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(sig, _on_signal)
    try:
        while proc.poll() is None:
            if stop:
                log.write(f"{stamp()} watchdog got signal {stop[0]}: resuming and terminating job\n")
                os.killpg(proc.pid, signal.SIGCONT)
                os.killpg(proc.pid, signal.SIGTERM)
                paused = False
                break
            raw = read_t()
            t, m, ac = thermal.update(raw), read_m(), read_ac()
            low_mem = m is not None and m < MEM_PAUSE_MB
            low_mem_since = (low_mem_since or time.time()) if (paused and low_mem) else None
            if low_mem_since is not None and time.time() - low_mem_since > mem_stall_s:
                log.write(f"{stamp()} memory stalled ({m:.0f} MB for {mem_stall_s:.0f}s): terminating job\n")
                os.killpg(proc.pid, signal.SIGCONT)
                os.killpg(proc.pid, signal.SIGTERM)
                paused = False
                break
            if m is not None and m < MEM_KILL_MB:
                log.write(f"{stamp()} memory critical ({m:.0f} MB): terminating\n")
                os.killpg(proc.pid, signal.SIGTERM)
                break
            action = decide(paused, t, m, ac)
            if action == "pause":
                os.killpg(proc.pid, signal.SIGSTOP)
                paused = True
                log.write(f"{stamp()} PAUSE temp_avg={t:.1f} raw={raw} mem={m:.0f} ac={ac}\n" if t is not None else f"{stamp()} PAUSE mem={m} ac={ac}\n")
            elif action == "resume":
                os.killpg(proc.pid, signal.SIGCONT)
                paused = False
                log.write(f"{stamp()} RESUME temp_avg={t} raw={raw} mem={m} ac={ac}\n")
            for _ in range(max(1, int(poll_s / 0.1))):
                if stop or proc.poll() is not None:
                    break
                time.sleep(0.1)
    finally:
        if paused and proc.poll() is None:
            os.killpg(proc.pid, signal.SIGCONT)
    rc = proc.wait()
    log.write(f"{stamp()} exit {rc}\n")
    if stop:
        # We were told to stop: propagate it so callers (e.g. the run queue) don't retry the job.
        raise SystemExit(128 + stop[0])
    return rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="runs/watchdog.log")
    ap.add_argument("cmd", nargs=argparse.REMAINDER)
    a = ap.parse_args()
    cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
    sys.exit(run(cmd, a.log))


if __name__ == "__main__":
    main()
