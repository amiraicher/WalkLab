---
name: python-coding-standards
description: Strict python coding standards for the WalkingLab project. Must be followed when writing or modifying any backend or UI Python code.
---

# Python Coding Standards

To ensure long-term maintainability and readability of the WalkingLab project, all agents must strictly adhere to the following coding standards:

## 1. Code Clarity & PEP 8
- **Clarity Over Cleverness:** Write code that is easy to read and understand. Avoid overly complex one-liners if a multi-line loop is more readable.
- **PEP 8 Compliance:** Strictly follow the PEP 8 style guide for Python code. This includes correct indentation (4 spaces), line length limits, and appropriate whitespace around operators.

## 2. Docstrings (NumPy Style)
- **Mandatory Documentation:** All modules, classes, and public functions/methods must have a docstring.
- **Format:** Use the strict **NumPy docstring standard**. 
- *Example:*
  ```python
  def calculate_velocity(positions, timestamps):
      """
      Calculates the velocity between consecutive positions.

      Parameters
      ----------
      positions : numpy.ndarray
          An N x 2 array of (x, y) coordinates.
      timestamps : numpy.ndarray
          An array of length N containing timestamps in seconds.

      Returns
      -------
      numpy.ndarray
          An (N-1) x 2 array of velocities.
      """
  ```

## 3. DRY Principle (Don't Repeat Yourself)
- **Reuse Before You Write:** Never duplicate logic. If you find yourself writing the same logic in multiple places (e.g., in `pose.py` and `annotation.py`), refactor it into a shared utility function in `walkinglab/backend/` or a shared UI component in `walkinglab/ui/components/`.

## 4. Naming Conventions
- **Meaningful Names:** Use highly descriptive, intention-revealing names for variables, functions, and classes. 
- Avoid single-letter variables (except for standard math iterators like `i`, `j`, `x`, `y`).
- **Standard Python Cases:**
  - Variables and functions: `snake_case` (e.g., `frame_count`, `process_video()`)
  - Classes: `PascalCase` (e.g., `VideoTracker`, `PoseEstimator`)
  - Constants: `UPPER_SNAKE_CASE` (e.g., `MAX_FRAME_RATE`)
