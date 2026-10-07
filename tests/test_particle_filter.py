import numpy as np
from pomdp_pick.belief.particle_filter import BeliefFilter
from pomdp_pick.sensing.camera import sigma_at, yaw_sigma, wrap_pi

def _run(seed, sigma_ep=0.010, steps=25, true=(0.2, 0.05, 1.4)):
    rng = np.random.default_rng(seed)
    pf = BeliefFilter(rng=np.random.default_rng(seed + 1))
    s = sigma_at(0.08, sigma_ep)
    z = np.array(true) + rng.normal(0, [s, s, yaw_sigma(s)])
    pf.reset_from_reading(z, 0.08)
    for _ in range(steps):
        pf.predict()
        z = np.array(true) + rng.normal(0, [s, s, yaw_sigma(s)])
        z[2] = wrap_pi(z[2])
        pf.update(z, 0.08)
    return pf

def test_deterministic_given_seed():
    a, b = _run(0).summary(), _run(0).summary()
    assert np.allclose(a["mu"], b["mu"]) and np.allclose(a["std"], b["std"])

def test_converges_and_handles_pi_wrap():
    pf = _run(1, true=(0.2, 0.05, 1.5))          # near +π/2: readings wrap to ~-π/2
    s = pf.summary()
    assert np.hypot(s["mu"][0] - 0.2, s["mu"][1] - 0.05) < 0.006
    assert abs(wrap_pi(s["mu"][2] - 1.5)) < 0.15
    assert s["std"][0] < 0.008

def test_estimates_noise_level():
    lo = _run(2, sigma_ep=0.003, steps=40).summary()["sigma_ep"]
    hi = _run(2, sigma_ep=0.025, steps=40).summary()["sigma_ep"]
    assert lo < 0.008 < hi

def test_prior_reset_spreads_over_workspace():
    pf = BeliefFilter(rng=np.random.default_rng(0))
    pf.reset_from_prior((0.0, 0.0), (0.13, 0.23), (-1.0, 1.0))
    s = pf.summary()
    assert s["std"][0] > 0.02 and np.all(np.isfinite(s["mu"]))
