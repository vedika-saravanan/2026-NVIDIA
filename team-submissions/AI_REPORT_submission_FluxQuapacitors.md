> **Note to Students:**  
> The questions and examples provided in the specific sections below are **prompts to guide your thinking**, not a rigid checklist.  
> * **Adaptability:** If a specific question doesn't fit your strategy, you may skip or adapt it.  
> * **Depth:** You are encouraged to go beyond these examples. If there are other critical technical details relevant to your specific approach, please include them.  
> * **Goal:** The objective is to convince the reader that you have employed AI agents in a thoughtful way.

---

## 1. The Workflow

We used multiple AI tools in a complementary way, each for a clearly scoped task rather than relying on a single agent for everything.

- **Gemini** and **CODA** were used to assist with drafting parts of the hybrid quantum–classical workflow in **Qiskit**, and for translating that workflow into **CUDA-Q**.
- **ChatGPT** was primarily used for:
  - Writing and refining project documentation
  - Explaining and restructuring code logic
  - Assisting with cross-framework translation when moving from Qiskit concepts to CUDA-Q constructs

By separating *algorithm design*, *framework translation*, and *documentation*, we reduced the risk of compounding errors and kept humans in the loop for all physics- and logic-critical decisions.

---

## 2. Verification Strategy

To validate AI-generated code and catch hallucinations or logical mistakes, we relied on **known physical ground-truth cases** rather than blindly trusting outputs.

- We explicitly designed a **control test cell** with problem size `n = 7`, where the **lowest energy state is analytically known to be 3**.
- This control case acted as a **unit test for the full workflow**, including:
  - Circuit construction
  - Hybrid quantum–classical execution flow
  - Energy computation and result interpretation

If the control cell failed or produced incorrect results:
1. We revisited the algorithmic notes and physics assumptions.
2. We manually inspected the generated circuit and workflow.
3. Logical or structural errors introduced by AI were identified and corrected before scaling up.

This approach ensured that AI assistance accelerated development without becoming a source of silent correctness bugs.

---

## 3. The “Vibe” Log

### Win — Where AI Saved Us Hours
Translating the workflow from **Qiskit to CUDA-Q**.

We were not fluent enough in CUDA-Q to write a complex hybrid quantum-classical algorithm from scratch within the hackathon timeline. Having AI handle the bulk of the translation allowed us to focus on validating correctness and performance rather than syntax and API minutiae.

---

### Learn — Improving Prompting Strategy
Initially, we tried to *explain* our workflow step-by-step to ChatGPT.

We later switched to **directly sharing the notebook** we had written and asked the model to:
- Generate documentation
- Explain the workflow *with respect to the actual code*

This context-first approach dramatically improved accuracy and reduced back-and-forth clarification.

---

### Fail — Where AI Hallucinated (and How We Fixed It)
We shared an image of **Exercise 4** with CODA, and it hallucinated several **extra quantum gates** that did not compile.

Fix:
- We manually inspected the generated circuit.
- Identified the non-physical / non-compiling gates.
- Removed them and revalidated the circuit against our control tests.

This reinforced the importance of **manual circuit inspection** when using AI for quantum workflows.

---

### Context Dump — Thoughtful Prompting Example

One example of a prompt that reflects deliberate, context-aware usage:

> “Can you please tell me which Nvidia GPUs in the image would perform well compared to Nvidia RTX 4080 GPU?”

This prompt tied **visual input**, **hardware constraints**, and **performance comparison** into a single targeted question, enabling faster decision-making during resource selection.

---
