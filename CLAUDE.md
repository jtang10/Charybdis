# CLAUDE.md — Charybdis Project Configuration

Generic rules (communication style, code style, workflow, testing, task management,
security) live in `~/.claude/CLAUDE.md`. This file contains only Charybdis-specific
overrides and notes.

---

## Project Reference

The full project plan, all design decisions with rationale, architecture, and
phase-by-phase milestones are in `PLAN.md`. Read it at the start of any session
involving this project.

---

## Purpose

This project is coding practice for learning — not production software. Build slowly
and deliberately. Understand each piece before adding the next.

---

## Key Directories

- `include/charybdis/` — C++ MLIR dialect headers
- `lib/charybdis/` — C++ dialect and lowering pass implementations
- `python/charybdis/` — Python DSL frontend
- `test/` — Python correctness tests and MLIR FileCheck tests

---

## Stack / Languages

- C++ (MLIR dialect definition, lowering passes)
- Python (DSL frontend, runtime, tests)
- TableGen (MLIR op definitions)
- CMake (build system)
- LLVM 20 / MLIR (compiler infrastructure)

---

## Dev Commands

```bash
# Build LLVM 20 (one-time, ~60 min) — see PLAN.md for full flags

# Configure Charybdis (run from ~/Charybdis)
cmake -B build \
  -DMLIR_DIR=$HOME/llvm-20-install/lib/cmake/mlir \
  -DMLIR_ENABLE_BINDINGS_PYTHON=ON \
  -DCMAKE_BUILD_TYPE=Release

# Build
cmake --build build -j$(sysctl -n hw.logicalcpu)

# Run Python end-to-end test (no GPU required)
PYTHONPATH=build/python_packages python test/test_phase0.py

# Run FileCheck tests (requires llvm-lit in PATH)
llvm-lit test/lit/
```

---

## Architecture Notes

- See PLAN.md for the full architecture diagram and dialect lowering strategy.
- IR mnemonic is `kbd` (short), not `charybdis` — ops appear as `kbd.identity`, `kbd.tile`, etc.
- C++ namespace: `mlir::charybdis`. Python package: `charybdis`.
- Two lowering targets: `nvgpu`/`nvvm` for GPU, `linalg` for CPU/debug.
- The CPU path is a first-class target — it runs the same MLIR pipeline, not a Python shortcut.
- LLVM built separately with `$HOME/llvm-20-install`; not a submodule. Use `$HOME`, not `~`, in CMake `-D` flags.

---

## Code Comment Style

Comments should be written from the perspective of a GPU performance engineer who
understands both GPU architecture (warps, warpgroups, TMA, tensor cores, shared
memory, barriers) and AI compiler design (MLIR dialects, lowering passes, IR
structure). Explain *why* something is done the way it is, not just what it does.

---

## Workflow for New Features

Before implementing any significant new feature or phase:
1. Write up a mini-plan explaining what is being built, the key design choices, and alternatives considered.
2. Get confirmation before writing code.
3. Implement incrementally — one op, one pass, one test at a time.
4. Do not move to the next phase until the current phase's exit criterion is met.
