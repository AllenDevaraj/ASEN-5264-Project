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
    marker = tmp_path / "child.pid"
    child = f"import os,time; open(r'{marker}','w').write(str(os.getpid())); time.sleep(60)"
    w = subprocess.Popen([sys.executable, str(pathlib.Path(__file__).parents[1] / "tools/watchdog.py"),
                          "--log", str(tmp_path / "w.log"), "--", sys.executable, "-c", child])
    for _ in range(50):
        if marker.exists() and marker.read_text():
            break
        time.sleep(0.1)
    cpid = int(marker.read_text())
    os.kill(cpid, signal.SIGSTOP)          # simulate a paused job
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
