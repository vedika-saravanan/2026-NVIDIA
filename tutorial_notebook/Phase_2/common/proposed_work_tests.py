import numpy as np
import pytest

import cudaq

from Phase_2.common.proposed_work import (
    compute_labs_energy,
    bitstring_to_spins,
    spins_to_bitstring,
    MemeticTabuSearch,
    MTSConfig,
    QuantumEnhancedMTS_CUDA,
)

# ----------------------------
# Deterministic unit tests
# ----------------------------

def test_bitstring_spin_roundtrip():
    bs = "010011"
    spins = bitstring_to_spins(bs)
    bs2 = spins_to_bitstring(spins)
    assert bs2 == bs

def test_compute_labs_energy_known_values():
    # N=3
    # [1,1,1]: C1 = 2, C2 = 1 => E = 4 + 1 = 5
    seq = np.array([1, 1, 1])
    assert compute_labs_energy(seq) == 5

    # [1,-1,1]: C1 = -2, C2 = 1 => E = 4 + 1 = 5 (not 1)
    # If you expected 1 earlier, that was a mismatch.
    seq = np.array([1, -1, 1])
    assert compute_labs_energy(seq) == 5

def test_compute_labs_energy_invariants():
    rng = np.random.default_rng(0)
    seq = rng.choice([1, -1], size=8)

    # Global flip doesn't change autocorrelation magnitudes
    assert compute_labs_energy(seq) == compute_labs_energy(-seq)

    # Reversal keeps the same set of pairwise products per lag
    assert compute_labs_energy(seq) == compute_labs_energy(seq[::-1])

def test_local_search_never_worsens_energy():
    n = 8
    cfg = MTSConfig(tabu_tenure=3, max_iterations=10, population_size=6)
    mts = MemeticTabuSearch(n, cfg)

    start = np.array([1, -1, 1, -1, 1, -1, 1, -1])
    start_e = compute_labs_energy(start)

    best, best_e = mts.local_search(start)
    assert best_e <= start_e

def test_mts_optimize_returns_valid_shape_and_energy():
    n = 6
    cfg = MTSConfig(tabu_tenure=2, max_iterations=5, population_size=6, elite_size=2)
    mts = MemeticTabuSearch(n, cfg)

    # small random population
    rng = np.random.default_rng(1)
    pop = [rng.choice([1, -1], size=n) for _ in range(cfg.population_size)]

    best, best_e = mts.optimize(pop)
    assert isinstance(best_e, (int, np.integer))
    assert best.shape == (n,)
    assert set(best.tolist()).issubset({-1, 1})
    # energy should match compute_labs_energy
    assert best_e == compute_labs_energy(best)

# ----------------------------
# Mocked quantum tests (no hardware, no simulator dependency)
# ----------------------------

class FakeSampleResult:
    """Mimics cudaq.sample() return value enough for .items()."""
    def __init__(self, counts):
        self._counts = counts
    def items(self):
        return list(self._counts.items())

def test_generate_quantum_candidates_uses_top_counts(monkeypatch):
    # Force deterministic "quantum" results
    fake_counts = {
        "0000": 50,
        "1111": 30,
        "0101": 20,
    }

    def fake_sample(*args, **kwargs):
        return FakeSampleResult(fake_counts)

    monkeypatch.setattr(cudaq, "sample", fake_sample)

    qe = QuantumEnhancedMTS_CUDA(n=4, qaoa_layers=1, shots=100)
    cands = qe.generate_quantum_candidates([0.1], [0.1], num_candidates=2)

    assert len(cands) == 2
    # First candidate should correspond to "0000" => spins all +1
    assert np.all(cands[0] == np.array([1, 1, 1, 1]))
    # Second should correspond to "1111" => spins all -1
    assert np.all(cands[1] == np.array([-1, -1, -1, -1]))

def test_optimize_calls_mts_and_returns_energy(monkeypatch):
    # Mock cudaq.sample so optimize doesn't rely on quantum runtime at all
    fake_counts = {"0000": 100}

    def fake_sample(*args, **kwargs):
        return FakeSampleResult(fake_counts)

    monkeypatch.setattr(cudaq, "sample", fake_sample)

    qe = QuantumEnhancedMTS_CUDA(n=4, qaoa_layers=1, shots=100)
    best_seq, best_e = qe.optimize(gamma=[0.1], beta=[0.1], mts_config=MTSConfig(max_iterations=3, population_size=4))

    assert best_seq.shape == (4,)
    assert best_e == compute_labs_energy(best_seq)


def test_kernel_is_invoked_via_cudaq_sample(monkeypatch):
    from test_dir.proposed_work import labs_qaoa

    calls = {}

    def fake_sample(kernel, *args, **kwargs):
        calls["kernel"] = kernel
        return type(
            "FakeResult",
            (),
            {"items": lambda self: [("000", 10)]}
        )()

    monkeypatch.setattr("cudaq.sample", fake_sample)

    qe = QuantumEnhancedMTS_CUDA(n=3, qaoa_layers=1, shots=10)
    candidates = qe.generate_quantum_candidates([0.1], [0.2], num_candidates=1)

    assert calls["kernel"] is labs_qaoa

    assert candidates[0].shape == (3,)
