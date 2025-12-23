# Copyright 2025 The zk_dtypes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python3
"""Compare notebook outputs to ensure they match the expected outputs.

This script compares the outputs of an executed notebook with the original
notebook to detect any changes in execution results.
"""

import json
import sys
from pathlib import Path


def normalize_output(output):
  """Normalize output for comparison by removing execution metadata."""
  if isinstance(output, dict):
    # Create a copy without execution metadata
    normalized = {}
    for key, value in output.items():
      # Skip execution metadata that may vary between runs
      if key in ("execution_count", "metadata"):
        continue
      # Recursively normalize nested structures
      if isinstance(value, dict):
        normalized[key] = normalize_output(value)
      elif isinstance(value, list):
        normalized[key] = [
            normalize_output(item) if isinstance(item, dict) else item
            for item in value
        ]
      else:
        normalized[key] = value
    return normalized
  return output


def compare_outputs(original_outputs, executed_outputs):
  """Compare outputs from two notebooks."""
  differences = []

  # Compare output lists
  if len(original_outputs) != len(executed_outputs):
    differences.append(
        f"Output count mismatch: original has {len(original_outputs)}, "
        f"executed has {len(executed_outputs)}"
    )
    # Still compare what we can
    min_len = min(len(original_outputs), len(executed_outputs))
    original_outputs = original_outputs[:min_len]
    executed_outputs = executed_outputs[:min_len]

  for i, (orig_out, exec_out) in enumerate(
      zip(original_outputs, executed_outputs)
  ):
    orig_norm = normalize_output(orig_out)
    exec_norm = normalize_output(exec_out)

    if orig_norm != exec_norm:
      # Try to show a more readable diff
      orig_str = json.dumps(orig_norm, indent=2, ensure_ascii=False)
      exec_str = json.dumps(exec_norm, indent=2, ensure_ascii=False)

      # If outputs are very long, truncate them
      max_len = 500
      if len(orig_str) > max_len:
        orig_str = orig_str[:max_len] + "... (truncated)"
      if len(exec_str) > max_len:
        exec_str = exec_str[:max_len] + "... (truncated)"

      differences.append(
          f"Output {i} differs:\n"
          f"  Original: {orig_str}\n"
          f"  Executed: {exec_str}"
      )

  return differences


def compare_notebooks(original_path, executed_path):
  """Compare two notebooks cell by cell."""
  with open(original_path, "r", encoding="utf-8") as f:
    original = json.load(f)

  with open(executed_path, "r", encoding="utf-8") as f:
    executed = json.load(f)

  differences = []
  original_cells = original.get("cells", [])
  executed_cells = executed.get("cells", [])

  if len(original_cells) != len(executed_cells):
    differences.append(
        f"Cell count mismatch: original has {len(original_cells)}, "
        f"executed has {len(executed_cells)}"
    )
    return differences

  for i, (orig_cell, exec_cell) in enumerate(
      zip(original_cells, executed_cells)
  ):
    # Only compare outputs for code cells
    if orig_cell.get("cell_type") != "code":
      continue

    orig_outputs = orig_cell.get("outputs", [])
    exec_outputs = exec_cell.get("outputs", [])

    cell_diffs = compare_outputs(orig_outputs, exec_outputs)
    if cell_diffs:
      differences.append(
          f"Cell {i} ({orig_cell.get('source', [''])[0][:50]}...):"
      )
      differences.extend(f"  {diff}" for diff in cell_diffs)

  return differences


def main():
  """Main entry point."""
  if len(sys.argv) != 3:
    print(
        "Usage: check_notebook_outputs.py <original_notebook> <executed_notebook>"
    )
    sys.exit(1)

  original_path = Path(sys.argv[1])
  executed_path = Path(sys.argv[2])

  if not original_path.exists():
    print(f"Error: Original notebook not found: {original_path}")
    sys.exit(1)

  if not executed_path.exists():
    print(f"Error: Executed notebook not found: {executed_path}")
    sys.exit(1)

  differences = compare_notebooks(original_path, executed_path)

  if differences:
    print("ERROR: Notebook outputs differ from expected outputs:")
    print("\n".join(differences))
    sys.exit(1)
  else:
    print("SUCCESS: All notebook outputs match expected outputs.")
    sys.exit(0)


if __name__ == "__main__":
  main()
