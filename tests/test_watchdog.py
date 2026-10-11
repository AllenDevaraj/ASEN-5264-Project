import importlib.util, pathlib
spec = importlib.util.spec_from_file_location("watchdog", pathlib.Path(__file__).parents[1] / "tools/watchdog.py")
wd = importlib.util.module_from_spec(spec); spec.loader.exec_module(wd)

def test_pause_on_heat():
    assert wd.decide(False, 86.0, 6000, True) == "pause"

def test_resume_only_below_hysteresis():
    assert wd.decide(True, 80.0, 6000, True) == "none"
    assert wd.decide(True, 74.0, 6000, True) == "resume"

def test_pause_on_low_memory_and_battery():
    assert wd.decide(False, 60.0, 1500, True) == "pause"
    assert wd.decide(False, 60.0, 6000, False) == "pause"
    assert wd.decide(True, 60.0, 6000, False) == "none"

def test_unknown_sensors_are_safe():
    assert wd.decide(False, None, None, None) == "none"
    assert wd.decide(True, None, None, None) == "resume"

def test_child_runs_to_completion(tmp_path):
    rc = wd.run(["python3", "-c", "print('ok')"], log_path=tmp_path / "w.log", poll_s=0.2)
    assert rc == 0
    assert "exit 0" in (tmp_path / "w.log").read_text()

def test_terminating_watchdog_terminates_paused_child(tmp_path):
    import os, signal, subprocess, sys, time
    # The child pauses itself before it can do anything, as the real watchdog may do on a hot or
    # low-memory machine, so the child pid comes from /proc rather than from the child.
    child = "import os,signal,time; os.kill(os.getpid(), signal.SIGSTOP); time.sleep(60)"
    w = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).parents[1] / "tools/watchdog.py"),
                          "--log", str(tmp_path / "w.log"), "--", sys.executable, "-c", child])
    children = pathlib.Path(f"/proc/{w.pid}/task/{w.pid}/children")
    for _ in range(200):
        if children.read_text().split():
            break
        time.sleep(0.05)
    cpid = int(children.read_text().split()[0])
    os.kill(cpid, signal.SIGSTOP)          # simulate a paused job (idempotent if already stopped)
    w.terminate()
    w.wait(timeout=20)
    time.sleep(0.5)
    try:
        os.kill(cpid, 0)
        alive = open(f"/proc/{cpid}/stat").read().split()[2] != "Z"
    except (ProcessLookupError, FileNotFoundError):
        alive = False
    assert not alive

def test_thermal_filter_ignores_turbo_spikes_but_not_sustained_heat():
    f = wd.ThermalFilter(window=6)
    for t in [60, 62, 61, 63, 60]:
        f.update(t)
    assert f.update(92) < wd.TEMP_PAUSE_C          # brief turbo spike: average stays low
    g = wd.ThermalFilter(window=6)
    for t in [88] * 6:
        eff = g.update(t)
    assert eff > wd.TEMP_PAUSE_C                   # sustained heat pauses
    h = wd.ThermalFilter(window=6)
    h.update(60)
    assert h.update(97) >= 97                      # above the hard cap: react immediately
    assert wd.ThermalFilter().update(None) is None
