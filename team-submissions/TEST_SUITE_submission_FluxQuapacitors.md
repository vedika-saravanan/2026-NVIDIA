# Test Suite and Verification Strategy

This document describes the verification strategy used to validate the correctness and robustness of our hybrid quantum–classical implementation. It also references the test code and coverage artifacts included in this repository.

---

## Overview

Our verification strategy is designed around the following principles:

- Deterministic logic should be validated using automated unit tests
- Quantum runtime execution should not be required for unit testing
- Integration points between quantum and classical components should be verified in a backend-independent manner
- Code coverage should reflect meaningful, testable logic rather than JIT-compiled runtime internals

To achieve this, we separate **shared algorithmic logic**, **automated tests**, and **experimental benchmarking**.

---

## Code Organization

The core algorithm and hybrid quantum–classical logic shared by both CPU and GPU workflows is implemented in:

- [`Phase_2/common/proposed_work.py`](../Phase_2/common/proposed_work.py)

Automated verification tests for this implementation are located in:

- [`Phase_2/common/proposed_work_tests.py`](../Phase_2/common/proposed_work_tests.py)

CPU- and GPU-specific execution and benchmarking are performed in Jupyter notebooks under:

Phase_2/CPU/Notebooks/
Phase_2/GPU/Notebooks/


This structure ensures that deterministic unit testing is clearly isolated from experimental evaluation and hardware-dependent execution.

---

## Unit Testing Strategy

The unit test suite focuses on deterministic components that can be rigorously verified without relying on quantum hardware or simulators. The following aspects are covered:

### LABS Energy Computation
The LABS energy function is tested using analytically tractable small sequences with known aperiodic autocorrelation values. These tests ensure numerical correctness and serve as a foundation for validating higher-level optimization logic.

### Algorithmic Invariants
We validate fundamental invariants of the LABS objective, including:
- Invariance under global spin flips
- Invariance under sequence reversal

These properties are intrinsic to the problem formulation and provide strong correctness guarantees beyond simple input–output testing.

### Data Conversion Utilities
Bitstring-to-spin and spin-to-bitstring conversion utilities are tested to ensure consistent and reversible interpretation of quantum measurement results.

### Classical Optimization Logic
The Memetic Tabu Search implementation is tested to verify that:
- Local search steps do not increase energy
- Population evolution returns valid spin sequences
- Returned energies are consistent with the LABS energy function

All unit tests are fully deterministic, fast to execute, and backend-independent.

---

## Mocked Quantum Integration Tests

CUDA-Q kernels are JIT-compiled runtime objects and cannot be meaningfully unit tested line-by-line using standard Python coverage tools. To verify correct integration without relying on real hardware or simulators, we mock the `cudaq.sample` interface in unit tests.

This allows us to:
- Verify that the correct CUDA-Q kernel object is passed to the quantum runtime
- Validate correct wiring of parameters (problem size, QAOA depth, angles, and shot count)
- Confirm deterministic downstream interpretation of quantum samples into candidate solutions

By mocking the quantum runtime, we ensure stable, reproducible tests while still validating the hybrid quantum–classical workflow.

---

## Code Coverage

Test coverage is measured using `pytest-cov`. The automated test suite achieves **86% statement coverage** for the shared implementation in `proposed_work.py`.

Coverage artifacts include:
- Terminal coverage output generated with `--cov-report=term-missing`
- An HTML coverage report generated with `--cov-report=html`

The uncovered lines correspond primarily to:
- CUDA-Q kernel, which are JIT-compiled and not traceable by Python coverage tools
- Execution-only code paths used for experimental benchmarking

Coverage therefore focuses on deterministic and verifiable components of the implementation.

---

## Coverage Evidence

The following images provide visual evidence of the coverage results:

- HTML coverage summary showing overall coverage percentage  
  ![HTML Coverage Report](../Phase_2/common/coverage_report_summary.png)

- Line-level coverage highlighting covered and uncovered regions  
  ![Line Coverage View](../Phase_2/common/coverage_report_lines.png)

These images are generated directly from `coverage.py` and reflect the automated test results.

---

## Test Design Rationale

The test cases were selected to maximize meaningful coverage by prioritizing:
- Deterministic logic with well-defined expected outcomes
- Fundamental algorithmic invariants
- Integration boundaries between quantum and classical components

This verification strategy provides high confidence in the correctness of the implementation while respecting the practical constraints of quantum runtime execution.

---

## Summary

All test code is included in this repository, and coverage artifacts are provided to demonstrate the extent and intent of the verification effort. The combination of deterministic unit tests, mocked quantum integration tests, and coverage analysis ensures robust and reproducible validation of the proposed approach.
