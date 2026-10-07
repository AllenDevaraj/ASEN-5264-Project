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
