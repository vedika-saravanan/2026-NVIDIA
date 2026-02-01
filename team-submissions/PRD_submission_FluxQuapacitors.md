# Product Requirements Document (PRD)

**Project Name:** QE-MTS for LABS 
**Team Name:** FluxQuapacitors
**GitHub Repository:** (https://github.com/vedika-saravanan/2026-NVIDIA/blob/main/tutorial_notebook/01_quantum_enhanced_optimization_LABS_FINAL.ipynb)

---


## 1. Team Roles & Responsibilities [You can DM the judges this information instead of including it in the repository]

| Role | Name | GitHub Handle | Discord Handle
| :--- | :--- | :--- | :--- |
| **Project Lead** (Architect) | Vedika Saravanan | https://github.com/vedika-saravanan | @vedikasaravanan |
| **GPU Acceleration PIC** (Builder) | Lim Wee Keat | https://github.com/WEEKEAT-LIM | @wee_keat |
| **Quality Assurance PIC** (Verifier) | Gabriel Ortega, Yash Singh | https://github.com/yiiyoo, https://github.com/git-Yassh | @yiiyo, @yash_goat |
| **Technical Marketing PIC** (Storyteller) | Travis Martin | [@handle] | @rifleexpertusmc |

---

## 2. The Architecture
**Owner:** Project Lead

### Choice of Quantum Algorithm
* **Algorithm:** 
    Quantum Approximate Optimization Algorithm (QAOA) with an explicit LABS cost Hamiltonian and a standard transverse-field mixer.

* **Motivation:** [Why this algorithm? Connect it to the problem structure or learning goals.]
    - We chose QAOA because the LABS problem admits a natural mapping to an Ising-style Hamiltonian containing both two-body and higher-order interaction terms. QAOA allows these problem-specific correlations to be encoded directly into the cost unitary while keeping circuit depth shallow and controllable.
    - In our implementation, the cost Hamiltonian is constructed explicitly from the LABS autocorrelation structure, including both two-body and four-body terms, and is decomposed into elementary gates compatible with CUDA-Q. Rather than using QAOA as a standalone optimizer, we use it as a quantum preprocessing step to bias sampling toward low-energy regions of the solution space.
    - This design aligns well with near-term constraints: shallow QAOA circuits can be efficiently simulated on GPUs, and the resulting quantum samples can meaningfully improve the initialization of a strong classical solver. The approach also supports incremental scaling, allowing us to increase circuit depth or refine parameter choices as problem size grows, while preserving interpretability and hybrid performance.
   

### Literature Review
* **Reference:** "Evidence of Scaling Advantage for the Quantum Approximate Optimization Algorithm on a Classically Intractable Problem,” Ruslan Shaydulin et al., Sci. Adv. Link: https://arxiv.org/abs/2308.02342"
* **Relevance:** [How does this paper support your plan?]
    * This work performs an extensive numerical investigation of QAOA applied directly to the LABS problem, showing that QAOA with fixed parameters can exhibit better empirical scaling than classical exact solvers for moderately sized instances of LABS. In particular, the authors observe improved time-to-solution behavior for QAOA compared to branch-and-bound and provide evidence that QAOA can act as a useful algorithmic component even when executed on ideal quantum computers. This supports our choice of QAOA as a starting point for quantum preprocessing, as it provides precedent for QAOA’s utility on the same problem domain we target.

* **Reference:** "Scaling Advantage with Quantum-Enhanced Memetic Tabu Search for LABS,” Alejandro Gomez Cadavid et al., 2025
Link: https://arxiv.org/abs/2511.04553"

* **Relevance:** [How does this paper support your plan?]
    * This recent result describes a hybrid algorithm — quantum-enhanced memetic tabu search (QE-MTS) that combines quantum sampling with classical tabu search and demonstrates improved scaling performance on the LABS problem compared to both pure classical heuristics and QAOA alone. The findings directly align with our hybrid approach, validating that quantum-seeded classical optimization can yield superior empirical scaling and motivating our workflow of using QAOA-generated samples to initialize a strong classical metaheuristic.

---

## 3. The Acceleration Strategy
**Owner:** GPU Acceleration PIC

### Quantum Acceleration (CUDA-Q)
* **Strategy:** [How will you use the GPU for the quantum part?]

    We use CUDA-Q to accelerate the quantum component by simulating QAOA circuits directly on NVIDIA GPUs. The QAOA cost Hamiltonian for LABS includes both two-body and four-body interaction terms, which are manually decomposed into CNOT ladders and single-qubit rotations to ensure compatibility with CUDA-Q backends.
    GPU acceleration is primarily leveraged for high-shot-count sampling of shallow QAOA circuits, allowing us to efficiently generate large batches of candidate bitstrings. These samples are ranked by frequency and converted into spin configurations that seed the classical Memetic Tabu Search. As problem size grows, GPU-accelerated simulation enables us to scale shot counts and circuit repetitions beyond what is practical on CPU alone.
 

### Classical Acceleration (MTS)
* **Strategy:** [The classical search has many opportuntities for GPU acceleration. What will you chose to do?]
    
    The classical Memetic Tabu Search is currently implemented on CPU to prioritize correctness, transparency, and ease of debugging. However, the algorithm exposes several clear opportunities for GPU acceleration. In particular, LABS energy evaluation dominates runtime during neighborhood exploration and population evolution.
    As a next step, we plan to batch-evaluate energy computations for multiple candidate sequences simultaneously using GPU-accelerated array operations (e.g., CuPy or custom CUDA kernels). This would allow parallel evaluation of mutation and crossover candidates, reducing the per-generation cost of tabu search while preserving identical algorithmic behavior.

### Hardware Targets
* **Dev Environment:**     As a next step, we plan to batch-evaluate energy computations for multiple candidate sequences simultaneously using GPU-accelerated array operations (e.g., CuPy or custom CUDA kernels). This would allow parallel evaluation of mutation and crossover candidates, reducing the per-generation cost of tabu search while preserving identical algorithmic behavior.

* **Production Environment:** NVIDIA L4 GPU (via Brev) for scalable CUDA-Q simulation and sampling, with the option to use higher-end GPUs (e.g., A100-80GB) for final large-N benchmarking and stress testing.


---

## 4. The Verification Plan
**Owner:** Quality Assurance PIC

### Unit Testing Strategy
* **Framework:** We use a combination of pytest and inline assertion-based tests embedded in the Jupyter notebook. Core logic (energy evaluation, interaction generation, tabu search, and helper utilities) is written in a testable, pure-Python form and validated using pytest-style test functions. This allows tests to be executed both inside the notebook during development and externally as part of a scripted validation run.

* **Test Organization:**
Deterministic unit tests for LABS energy computation, symmetry properties, and interaction indexing
Functional tests for classical MTS components (local search, tabu behavior, crossover, mutation)
Integration tests that validate quantum-generated bitstrings using the same classical energy function used by MTS
* **AI Hallucination Guardrails:** [How do you know the AI code is right?]

Any AI-assisted code (quantum kernels or classical refactors) must pass the full test suite before being integrated. In particular, AI-generated CUDA-Q kernels are required to:

    1. Execute without runtime errors on the CUDA-Q simulator
    2. Produce valid bitstrings of the correct length
    3. Yield energies consistent with independent classical evaluation
    Code that violates known LABS invariants or produces inconsistent results is rejected and rewritten.

### Core Correctness Checks
* **Check 1 (Symmetry):** [Describe a specific physics check]

    LABS energies are invariant under global bit-flip and sequence reversal. Using pytest, we assert for randomly generated sequences S:

    ```
    assert energy(S) == energy(-S)
    assert energy(S) == energy(S[::-1])

    ```
    These tests validate correct indexing, sign handling, and Hamiltonian construction.

* **Check 2 (Ground Truth):**
    
    For small problem sizes (N=3 andN=4), energies are computed analytically and compared against the implemented energy function. For example:

    ```
    assert energy(np.array([1, 1, -1])) == 1

    ```
Quantum-generated bitstrings are converted to spin configurations and evaluated using the same classical energy function, ensuring consistency between quantum sampling and classical scoring.
---

## 5. Execution Strategy & Success Metrics
**Owner:** Technical Marketing PIC

### Agentic Workflow
* **Plan:** [How will you orchestrate your tools?]

    Development is organized as a structured human-in-the-loop workflow rather than fully autonomous agent execution. All code is written and iterated in VSCode and Jupyter notebooks, with CUDA-Q documentation referenced directly to avoid API misuse.
    Quantum kernels, classical solvers, and integration logic are developed independently and validated through unit and integration tests before being combined. When changes are required, error traces and test failures are reviewed explicitly and used to guide refactoring, ensuring that debugging decisions remain interpretable and grounded in domain knowledge rather than trial-and-error automation.

### Success Metrics

* **Metric 1 (Approximation):**
Quantum-generated initial populations achieve lower average LABS energy than randomly initialized populations for the same problem size.
* **Metric 2 (Speedup):** 
QAOA-seeded Memetic Tabu Search consistently achieves equal or better final energies compared to purely classical MTS under identical computational budgets.
* **Metric 3 (Scale):** 
Successful execution and sampling of QAOA circuits and hybrid optimization for problem sizes up to  N≥20, with a clear path toward larger instances as GPU resources allow.



### Visualization Plan
* **Plot 1:** 
Convergence curves showing LABS energy versus MTS iteration count
* **Plot 2:**
Distribution of initial population energies for quantum-generated samples versus randomly generated samples.


---

## 6. Resource Management Plan
**Owner:** GPU Acceleration PIC 

* **Plan:** [How will you avoid burning all your credits?]

    To minimize unnecessary GPU usage, all algorithm development, debugging, and verification are performed on CPU-based environments first. This includes implementing the LABS energy function, validating the Memetic Tabu Search logic, and verifying quantum circuit structure using low-shot simulations.
    GPU resources are introduced only after correctness is established. Initial GPU testing is conducted on lower-cost NVIDIA L4 instances to validate CUDA-Q execution and sampling behavior. High-end GPU instances (e.g., A100) are reserved strictly for short, scheduled benchmarking runs at larger problem sizes or higher shot counts.
    GPU usage is tightly controlled through manual oversight. The GPU Acceleration PIC is responsible for starting and stopping cloud instances as needed and ensuring that no instances remain active during idle periods.
