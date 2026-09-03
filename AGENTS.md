# Agent Operating Instructions & Workflow Protocol

## 1. Context Initialization (Mandatory First Step)
Before executing any prompt or writing code, the agent must:
1. **Read `General_Plan.md`:** Understand the master project goals, constraints, and architecture choices.
2. **Read or Initialize `Architecture.md`:** Check the current system architecture state. If `Architecture.md` does not exist, create it based on the current codebase state and `General_Plan.md`.

---

## 2. Architecture Living-Doc Rule (`Architecture.md`)
* The project architecture must **always** be documented in `Architecture.md`.
* **Rule:** Whenever a task modifies code structures, adds modules, or changes data flows, the agent **must** update `Architecture.md` as the final step before concluding the task.

---

## 3. Multi-Subagent Execution Protocol
For every user prompt requiring code changes, the agent must orchestrate a structured multi-subagent workflow divided into three phases:

### Phase 1: Planning & Delegation
1. Analyze the user request against `General_Plan.md` and `Architecture.md`.
2. Formulate a step-by-step implementation plan.
3. Spawn/assign tasks to two distinct specialized subagents:
   * **Coder Agent:** Responsible for code implementation.
   * **Tester / QA Agent:** Responsible for creating the test plan and verification scripts.

### Phase 2: Parallel/Sequential Execution
1. **The Coder Agent** implements the code changes in the repository following clean architecture rules and relevant project skills (e.g., pose guidelines, NiceGUI rules).
2. **The Tester Agent** simultaneously drafts a comprehensive test plan and writes automated test scripts/assertions covering the new functionality.

### Phase 3: Iterative Verification & Bug Fixing Loop
1. The Tester executes the verification plan against the newly written code.
2. **If tests fail or bugs are found:** 
   * The Tester provides precise error logs and failure reports back to the Coder.
   * The Coder applies fixes.
   * The Tester re-runs the tests.
3. This loop repeats iteratively until all tests pass successfully.
4. **Final Step:** Update `Architecture.md` to reflect any new structural changes introduced during the cycle.
---

## 4. Git & Committing Rule
* **Rule:** Never auto-commit changes to the git repository. When a task is complete, notify the user and ask them to perform the commit.
