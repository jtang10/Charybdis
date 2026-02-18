# Charybdis DSL — MVP Plan

## Project Overview

Charybdis is a Python-based DSL that compiles to MLIR and targets Nvidia B200 (Blackwell) GPUs at the warp/warpgroup level. It also supports a CPU debug path via MLIR's linalg dialect → numpy.

This project is coding practice — the goal is to learn GPU architecture and AI compiler construction by building something real, not to ship a product. The pace is deliberately slow and incremental. Each phase should be understood deeply before the next begins.

---

## Development Principles

**Build slowly.** Each phase introduces one new concept. Don't stack multiple new ideas in a single implementation step.

**Understand before advancing.** The exit criterion for each phase is not just "code compiles" — it's "I understand why it works."

**Write a mini-plan before every significant feature.** Before implementing any new op, lowering pass, or runtime component, write a short design note covering:
- What is being built and why it is needed at this point
- The key design choices and alternatives considered
- Any GPU architecture or compiler concepts that are central to the implementation

**Comment for the persona.** Code comments should reflect the perspective of a GPU performance engineer who understands both hardware (warp execution model, tensor cores, TMA, shared memory banks, barrier synchronization) and compiler design (MLIR's progressive lowering model, dialect layering, op semantics, type systems). Comments explain *why* the code is structured the way it is — the hardware or compiler reason behind each decision.

---

## Design Decisions

This section documents every significant design choice, the alternatives considered, and the rationale for what was selected. Where Triton or Mosaic GPU made the same or a different choice, that is noted explicitly.

---

### 1. MLIR Backend Dialect

**Choice: Hybrid — `nvgpu` dialect as the primary target, escape to `nvvm` for gaps.**

The three realistic options were:

**Option A: NVVM/PTX dialect (direct PTX codegen)**
Lower the entire IR to MLIR's `nvvm` dialect, which maps 1:1 to NVVM IR (Nvidia's flavor of LLVM IR), then emit PTX via the LLVM NVPTX backend.

- Pros: Maximum control. Every PTX instruction is explicitly expressed. Useful for squeezing out last-mile performance or targeting hardware features that aren't yet modeled in higher-level dialects.
- Cons: You must manually manage all warp-level semantics — barriers, predication, shared memory layout, warp synchronization. The amount of lowering code required is large. Every new B200 feature requires new manual handling.

**Option B: Generic `gpu` dialect + LLVM NVPTX backend**
Use MLIR's hardware-agnostic `gpu` dialect (`gpu.launch`, `gpu.thread_id`, etc.) and lower through LLVM IR.

- Pros: Portable — the `gpu` dialect doesn't commit to Nvidia, so upper layers are reusable for AMD or other targets in theory.
- Cons: The `gpu` dialect thinks in terms of threads and blocks, not warps or warpgroups. It is too coarse to express the warp-level tiling that Charybdis targets. You inevitably drop to `nvvm` intrinsics for anything involving tensor cores, TMA, or barriers, making the `gpu` dialect layer a thin and unhelpful indirection.

**Option C: `nvgpu` dialect (Hopper/Blackwell-specific)**
Use MLIR's `nvgpu` dialect, which models Hopper and Blackwell hardware features directly: `nvgpu.warpgroup.mma` for wgmma tensor core instructions, `nvgpu.tma.async.load/store` for the Tensor Memory Accelerator, and `nvgpu.mbarrier` for hardware-level producer/consumer synchronization.

- Pros: The dialect was designed precisely for the hardware Charybdis targets. First-class ops for wgmma, TMA, and mbarrier mean far less hand-written lowering code. Actively maintained upstream by the MLIR/NVIDIA team.
- Cons: Hopper-centric; some very new Blackwell-specific variants may not yet have dialect coverage. Sparser documentation than `nvvm`.

**Why this choice:** Mosaic GPU (Google's Hopper/Blackwell kernel DSL inside JAX) uses the `nvgpu` dialect as its primary lowering target and has proven it works for production matmul and attention kernels. Since Charybdis targets the same hardware with similar goals, this is the validated path. The hybrid approach — `nvgpu` where covered, `nvvm` intrinsics for gaps — avoids being blocked when `nvgpu` doesn't yet model a specific B200 feature.

Triton historically targeted `nvvm`/PTX directly and built its own warp-level abstractions from scratch. This gave Triton maximum control but required significant engineering effort to maintain and extend. Charybdis avoids that cost by starting with `nvgpu`.

---

### 2. Abstraction Level (How Explicit Users Are About Hardware Resources)

**Choice: Meet in the middle — users specify tile shapes and hints; the compiler decides register layout, shared memory allocation, and warp assignment.**

The three options were:

**Option A: High-level tiling only (fully automatic layout)**
Users specify shapes and ops; the compiler chooses everything — which warps handle which tile, how registers are laid out, how shared memory is organized.

- Pros: Simplest user-facing interface. Least expert knowledge required.
- Cons: The compiler must solve a hard layout optimization problem correctly for every kernel. For a new project this is impractical to get right immediately, and it limits the user's ability to express expert knowledge about the hardware.

**Option B: Fully explicit warp/warpgroup tiling**
Users manually annotate which warpgroup handles which tile, manually control shared memory allocation, and explicitly express all synchronization.

- Pros: The compiler's job is simple — it mostly just lowers what the user wrote. Maximum expert control.
- Cons: Shifts the hardware complexity burden entirely onto the user. Defeats the purpose of a DSL. Equivalent to writing PTX by hand.

**Option C: Meet in the middle**
Users specify tile sizes and can provide hints (e.g., number of warpgroups, pipeline stages). The compiler makes concrete layout decisions but exposes overrides for expert users who need them.

- Pros: Practical balance. The DSL is usable without deep hardware knowledge, but an expert can push performance further. This is also the most extensible design — you can start with simple heuristics in the compiler and improve them without changing the user-facing API.
- Cons: The compiler's heuristics must be good enough to produce correct and reasonably performant code, which takes iteration.

**Why this choice:** Both Triton and Mosaic GPU take versions of this approach. Triton lets users specify tile shapes (via `tl.constexpr`) and the compiler handles thread/warp mapping. Mosaic GPU similarly exposes tile granularity and warpgroup count while hiding register layout. Charybdis follows the same principle because it enables a usable MVP without requiring the compiler to solve the full auto-layout problem on day one.

---

### 3. Python DSL Syntax Style

**Choice: Decorator-based, Triton-style — Python functions annotated with `@kernel`, taking `Tile`-typed arguments.**

The three options were:

**Option A: Decorator-based (Triton-style)**
```python
@kernel
def vector_add(a: Tile[f32, 1024], b: Tile[f32, 1024], c: Tile[f32, 1024]):
    c[:] = a[:] + b[:]
```
The decorator intercepts the function, traces or compiles it, and produces an executable kernel object.

- Pros: Familiar to anyone who has used Triton, Numba, or JAX's `@jit`. Looks like ordinary Python, so editor tooling (autocomplete, type checkers) works reasonably well. Easy to explain to users.
- Cons: The Python inside the kernel body is not actually executed as Python — it is traced or parsed. This creates a semantic gap that can surprise users (e.g., Python control flow may not behave as expected unless handled carefully).

**Option B: Staged/traced (JAX-style)**
The kernel is a Python function that gets traced by running it with abstract values, building a graph of operations that is then compiled. This is how JAX's `jit` and XLA work.

- Pros: The user writes natural NumPy-like code. Tracing is a well-understood technique.
- Cons: Tracing has fundamental limitations — data-dependent control flow is hard to handle. Supporting dynamic shapes requires significant infrastructure (abstract interpretation, symbolic shapes). For a GPU DSL with fixed tile sizes, tracing adds complexity without clear benefit over the decorator approach.

**Option C: Builder API (explicit IR construction)**
```python
b = KernelBuilder()
a = b.load(ptr_a, shape=1024)
c = b.add(a, b.load(ptr_b, shape=1024))
b.store(c, ptr_c)
```
- Pros: No semantic gap — what you write is exactly what IR gets built. Fully inspectable.
- Cons: Verbose. Not Pythonic. High barrier to entry. Harder to read kernel logic at a glance.

**Why this choice:** Triton uses the decorator approach and it has proven to be the most productive interface for GPU kernel authorship. It is the most widely understood pattern in this domain. The semantic gap between "Python syntax" and "what actually runs" is manageable if the DSL is designed carefully (e.g., by restricting the supported subset and giving clear error messages when unsupported Python is used inside a kernel).

---

### 4. Python Binding Strategy

**Choice: Upstream MLIR Python bindings (`mlir.core`).**

The three options were:

**Option A: Upstream MLIR Python bindings**
MLIR ships its own Python bindings that expose `MLIRContext`, `Module`, `Block`, `Op`, `Type`, and related primitives. You extend these by registering your custom dialect's Python wrappers via MLIR's binding infrastructure.

- Pros: No custom binding code to write for the core MLIR machinery. Your dialect's Python API integrates cleanly with the rest of the MLIR Python ecosystem. Well-maintained upstream.
- Cons: The bindings are verbose and somewhat low-level — you still need to write per-op Python wrappers for your custom dialect ops.

**Option B: pybind11 (custom bindings)**
Write your own pybind11 wrappers around your C++ MLIR passes and dialect objects.

- Pros: Full control over the Python API surface. Can expose exactly the interface you want.
- Cons: Significant boilerplate. You reimplement what MLIR's bindings already provide. Maintenance burden grows as the dialect grows.

**Option C: nanobind (modern pybind11 alternative)**
Same idea as pybind11 but with faster compile times, smaller binaries, and a cleaner API. Used by Triton.

- Pros: Better developer experience than pybind11. Triton's adoption validates it for this use case.
- Cons: Triton uses nanobind because it has a fully custom compiler pipeline that doesn't reuse MLIR's binding infrastructure. For Charybdis, which does reuse MLIR's infrastructure, mixing nanobind with the upstream MLIR bindings creates unnecessary complexity.

**Why this choice:** Mosaic GPU uses the upstream MLIR Python bindings and extends them for its custom ops. Since Charybdis takes the same architectural approach as Mosaic GPU (custom dialect + lowering passes on top of MLIR), the same binding strategy applies. This avoids reinventing infrastructure that MLIR already provides.

---

### 5. CPU / Debug Fallback

**Choice: Lower to MLIR's `linalg` dialect, then execute via the MLIR execution engine (which JITs to native code), bridging results back to numpy arrays.**

The two options were:

**Option A: Pure Python interpreter (no MLIR involvement)**
When running in CPU mode, the `@kernel` decorator simply executes the kernel body as Python, substituting numpy operations for tile operations.

- Pros: Trivially simple to implement. Immediate results.
- Cons: The CPU path exercises almost none of the compiler stack. Bugs in the MLIR lowering pipeline, type system, or op semantics would not be caught by CPU tests. The two paths diverge and you lose confidence that the CPU result actually validates the GPU path.

**Option B: Lower to `linalg` dialect → MLIR execution engine → numpy**
The same MLIR pipeline runs for CPU targets, but instead of lowering to `nvgpu`/`nvvm`, ops are lowered to `linalg` generics, which are then lowered to loops and compiled to native code via MLIR's JIT execution engine. Results are read back as numpy arrays.

- Pros: The compiler stack — type checking, op semantics, lowering correctness — is exercised on every CPU test. A bug caught by a CPU test is a real compiler bug, not just a Python logic error. This also validates that the `charybdis` dialect ops have a well-defined semantics independent of the GPU backend.
- Cons: More implementation work. The linalg lowering pass must be written alongside the nvgpu pass.

**Why this choice:** The linalg path is more principled and maximizes the value of CPU tests. Triton does not have a strong CPU fallback; its debugging story is primarily GPU-side. Mosaic GPU similarly lacks a CPU fallback. Charybdis treats the CPU path as a first-class target from the start, which is particularly important given that a B200 machine is not available during early development.

---

### 6. Build System

**Choice: CMake, with LLVM 20 built separately and installed to a prefix that CMake finds via `find_package(MLIR)`.**

The three options were:

**Option A: CMake + LLVM as a git submodule**
Clone LLVM into `third_party/llvm-project` and build it as part of the Charybdis build.

- Pros: Fully self-contained. Exact LLVM version is pinned in the repo.
- Cons: LLVM is ~3GB of source and takes 30–90 minutes and ~50GB of disk to build. Rebuilding it on every clean checkout is impractical. CI becomes very slow.

**Option B: Use a pre-built LLVM (system package or pip-installed)**
Install LLVM 20 via the system package manager or via `pip install mlir` and point CMake at it.

- Pros: Fast to get started. No build time for LLVM.
- Cons: The pip-distributed MLIR packages often lag behind upstream and may not include headers or CMake config files needed for out-of-tree dialect development. System packages vary by distro and may not include the right targets.

**Option C: Build LLVM 20 separately, install to a local prefix, point CMake at it**
Build LLVM once with the required targets (`NVPTX`, Python bindings enabled), install to `~/llvm-20-install` or similar, and set `MLIR_DIR` when configuring Charybdis.

- Pros: Clean separation. LLVM is built once and reused across builds. Full control over build flags (optimized build, assertions on/off, which targets are included). This is how Triton and Mosaic GPU handle their LLVM dependency.
- Cons: Requires an upfront LLVM build (one-time cost).

**Why this choice:** Both Triton and Mosaic GPU build against a separately-compiled LLVM rather than bundling it as a submodule or relying on system packages. This is the standard approach for production out-of-tree MLIR projects. The one-time build cost is acceptable; the ongoing cost of a submodule build is not.

---

### 7. LLVM Version

**Choice: LLVM 20.**

The two options were:

**Option A: LLVM 19**
Stable, widely used. Most `nvgpu` and `nvvm` dialect ops for Hopper (H100) are present. Smaller chance of hitting upstream bugs.

- Cons: Some Blackwell-specific `nvgpu` op variants and new wgmma instruction forms may not be present. B200 is a newer target and LLVM 19's Blackwell coverage is incomplete.

**Option B: LLVM 20**
Latest stable release. Best coverage for Blackwell-specific features in the `nvgpu` and `nvvm` dialects. Active development means B200-specific intrinsics are more likely to be present.

- Cons: Slightly less battle-tested than LLVM 19. Possible upstream bugs in new features.

**Why this choice:** Charybdis explicitly targets B200. Using LLVM 19 and hitting a missing Blackwell intrinsic early in development would be a painful blocker. LLVM 20's improved Blackwell coverage outweighs the stability risk, especially since the project is starting fresh and is not constrained by an existing LLVM 19 dependency.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                Python DSL Frontend                   │
│  @kernel decorator  │  Tile[dtype, shape] type       │
│  Traces Python fn → builds charybdis MLIR ops       │
│  Uses upstream MLIR Python bindings (mlir.core)     │
└─────────────────────────────┬───────────────────────┘
                              │
                    charybdis dialect (C++ MLIR)
                    charybdis.tile_op, .warpgroup_mma, etc.
                              │
               ┌──────────────┴──────────────┐
               │                             │
        GPU target                    CPU/debug target
               │                             │
        nvgpu + nvvm                   linalg dialect
        + gpu dialect                        │
               │                        numpy (via
        LLVM NVPTX backend             MLIR execution engine
               │                        + memref → numpy)
             PTX/CUBIN                       │
               │                        Python numpy arrays
        CUDA driver API                 (correctness verified)
```

---

## Repository Layout

```
charybdis/
├── CMakeLists.txt
├── cmake/
│   └── LLVMConfig.cmake
├── include/
│   └── charybdis/
│       ├── Dialect/
│       │   ├── CharybdisDialect.h
│       │   ├── CharybdisOps.h
│       │   └── CharybdisTypes.h
│       └── Transforms/
├── lib/
│   └── charybdis/
│       ├── Dialect/
│       └── Transforms/
│           ├── CharybdisToNVGPU.cpp
│           └── CharybdisToLinalg.cpp
├── python/
│   └── charybdis/
│       ├── __init__.py
│       ├── kernel.py           # @kernel decorator + tracer
│       ├── types.py            # Tile type
│       ├── ops.py              # DSL ops (warpgroup_mma, reduce, etc.)
│       ├── runtime/
│       │   ├── cuda.py         # CUDA driver launch
│       │   └── numpy_rt.py     # CPU/debug runtime
│       └── _mlir_libs/         # Built .so lands here
├── test/
│   ├── vector_add.py
│   └── lit/                    # MLIR FileCheck tests
└── third_party/                # LLVM pointed to externally
```

---

## GPU Availability Notes

Development can proceed almost entirely on macOS until GPU execution is needed.

| Phase | GPU Required? | Notes |
|---|---|---|
| Phase 0 | No | Build system, dialect definition, PTX emission — all on macOS |
| Phase 1 (compiler) | No | `@kernel` tracing, lowering passes, PTX output — all on macOS |
| Phase 1 (runtime) | Yes (first time) | Actually launching and verifying the kernel on device |
| Phase 2+ | Same pattern | Compiler side on Mac; execution needs GPU |

**Strategy:** Two test modes from day one:
- `make test-cpu` — runs linalg → numpy path, fully on macOS
- `make test-gpu` — runs on CUDA device (CI or remote)

Use MLIR `lit` + `FileCheck` to assert correct PTX/MLIR structure without executing it. This covers ~80% of compiler validation on Mac. Use a cloud GPU instance (Lambda Labs, RunPod, CoreWeave) for execution testing when needed.

---

## Phases

Each phase begins with a mini-plan. Do not write code until the mini-plan is written and agreed on.

---

### Phase 0 — Build System & Bootstrapping

**Goal:** Compile a trivial MLIR program through the full stack to PTX. Establish that all the tooling works before writing any real compiler code.

**Naming:** IR mnemonic is `kbd` (e.g. `kbd.identity`). C++ namespace is `mlir::charybdis`. See PHASE0.md for the full checklist.

**Concepts introduced:** MLIR out-of-tree dialect structure, TableGen op definitions, CMake/LLVM integration, MLIR Python binding registration.

- [x] Mini-plan written and agreed (see PHASE0.md)
- [x] CMake setup: find LLVM 20 install, register `charybdis` dialect
- [x] TableGen: one dummy op (`kbd.identity`)
- [x] MLIR Python bindings: expose dialect to Python
- [x] Lowering pass: `convertCharybdisToLLVM` (identity → operand passthrough)
- [x] `charybdis-opt` tool for FileCheck tests
- [x] Tests: `test/lit/identity.mlir` (FileCheck) + `test/test_phase0.py` (Python)

**Exit criterion:** `llvm-lit test/lit/` and `python test/test_phase0.py` both pass on macOS without a GPU. I can explain what TableGen generated and how the Python bindings wired up.

---

### Phase 1 — vectorAdd (end-to-end, warp-level)

**Goal:** First real kernel through the full stack. This phase establishes every layer of the system — Python frontend, MLIR dialect, lowering passes, and runtime — for the simplest possible operation.

**Concepts introduced:** `@kernel` decorator and Python tracing, the `Tile` type, warp-level data partitioning (1024 elements / 32 lanes = 32 elements per thread), elementwise op lowering, the two-target lowering pipeline (GPU vs CPU), MLIR execution engine, memref ↔ numpy bridge.

**DSL surface:**
```python
@kernel
def vector_add(a: Tile[f32, 1024], b: Tile[f32, 1024], c: Tile[f32, 1024]):
    c[:] = a[:] + b[:]
```

**Tasks:**
- [ ] Mini-plan written and agreed (covering: how `@kernel` traces Python, what `charybdis.tile` looks like in MLIR, how warp partitioning maps to thread indexing in PTX)
- [ ] `Tile[dtype, shape]` Python type → `charybdis.tile` MLIR type
- [ ] `@kernel` decorator: traces Python → charybdis ops in MLIR
- [ ] `charybdis.tile_elementwise` op
- [ ] GPU lowering pass: `charybdis → gpu + arith + memref`
- [ ] CPU lowering pass: `charybdis → linalg`
- [ ] GPU runtime: PTX → CUBIN → kernel launch via `cuda-python`
- [ ] CPU runtime: MLIR execution engine + memref ↔ numpy bridge
- [ ] FileCheck tests for PTX structure
- [ ] Correctness test: GPU output matches numpy reference

**Exit criterion:** `vector_add.py` produces correct output on B200 and matches numpy on CPU. I can explain how a `Tile[f32, 1024]` becomes per-thread register storage in PTX.

---

### Phase 2 — Matmul (warpgroup + tensor cores)

**Goal:** Introduce warpgroup tiling, the wgmma tensor core instruction, and TMA async data movement. This is where the B200-specific hardware features enter the picture.

**Concepts introduced:** Warpgroup (4 warps = 128 threads acting as a unit), `wgmma` (Hopper/Blackwell's asynchronous warp-group matrix multiply-accumulate), TMA (Tensor Memory Accelerator — a hardware unit that performs bulk async copies between global and shared memory), `mbarrier` (hardware barrier for synchronizing TMA completion with warpgroup compute), shared memory allocation and layout.

**DSL surface:**
```python
@kernel(warpgroups=1)
def matmul(A: Tile[f16, (M, K)], B: Tile[f16, (K, N)], C: Tile[f32, (M, N)]):
    acc = warpgroup_mma(A, B)
    C[:] = acc
```

**Tasks:**
- [ ] Mini-plan written and agreed (covering: wgmma instruction shape constraints, TMA descriptor setup, mbarrier producer/consumer pattern, how nvgpu dialect models all three)
- [ ] `charybdis.warpgroup_mma` op → `nvgpu.warpgroup.mma`
- [ ] `charybdis.tma_load` op → `nvgpu.tma.async.load`
- [ ] Shared memory type + allocation in dialect
- [ ] `mbarrier` synchronization ops
- [ ] CPU lowering: `warpgroup_mma → linalg.matmul`
- [ ] Correctness check against cuBLAS

**Exit criterion:** FP16 matmul on B200 using wgmma + TMA, verified correct. I can explain the wgmma instruction's shape constraints and why TMA exists (bypassing the L2 cache and freeing the warpgroup from data movement).

---

### Phase 3 — Reduction

**Goal:** Introduce cross-warp communication. This is the first operation where threads must exchange values with each other, not just read/write independent memory locations.

**Concepts introduced:** Warp shuffle instructions (`shfl.sync` — exchange register values between threads in a warp without going through shared memory), warp-level tree reduction, cross-warp reduction via shared memory, predicated writes for the scalar output case.

**DSL surface:**
```python
@kernel
def sum_reduce(a: Tile[f32, 1024], out: Tile[f32, 1]):
    out[0] = reduce(a, op="sum")
```

**Tasks:**
- [ ] Mini-plan written and agreed (covering: warp shuffle mechanics, the log2(32)-step butterfly reduction pattern, why shuffles are faster than shared memory for intra-warp communication)
- [ ] `charybdis.reduce` op (sum, max, min)
- [ ] Warp-level reduce lowering via `nvvm.shfl.sync`
- [ ] Cross-warp reduce via shared memory
- [ ] CPU lowering: `linalg` reduction

**Exit criterion:** Correct reduction result on B200 and CPU. I can trace through the butterfly reduction pattern and explain when shuffle vs shared memory is preferred.

---

### Phase 4 — Scan

**Goal:** Prefix operations. Unlike reduction (many-to-one), scan is many-to-many — every output depends on all prior inputs. Introduces ordering within a warp.

**Concepts introduced:** Inclusive vs exclusive prefix scan, the Hillis-Steele parallel scan algorithm using warp shuffles, inter-warp scan coordination via shared memory, the distinction between associative and non-associative scan operators.

**Tasks:**
- [ ] Mini-plan written and agreed (covering: Hillis-Steele algorithm, why scan is harder than reduction, how to extend warp-level scan to multi-warp)
- [ ] `charybdis.scan` op (prefix sum, prefix max)
- [ ] Warp-level scan via shuffle instructions
- [ ] Inter-warp scan via shared memory

**Exit criterion:** Correct prefix sum on B200 and CPU. I can explain why scan requires O(log n) synchronization steps.

---

### Phase 5 — Attention

**Goal:** Compose all previous primitives into a FlashAttention-style fused kernel. This is the first "real" kernel that a production system would care about.

**Concepts introduced:** Online softmax (numerically stable, single-pass), the FlashAttention tiling strategy (process Q/K/V in tiles to stay in SRAM), producer/consumer warpgroup split (one warpgroup runs TMA, another runs wgmma), multi-stage pipelining (double-buffering to overlap data movement with compute).

**Tasks:**
- [ ] Mini-plan written and agreed (covering: FlashAttention algorithm, online softmax derivation, pipeline stages, warpgroup role assignment)
- [ ] Online softmax (FlashAttention-style numerically stable reduction)
- [ ] Compose `warpgroup_mma` + `reduce` + `warpgroup_mma`
- [ ] TMA prefetch pipelining: overlap data movement with compute
- [ ] Correctness check against reference attention implementation

**Exit criterion:** Correct attention output on B200 and CPU. I can explain the online softmax recurrence and why the FlashAttention tiling avoids materializing the full attention matrix.

---

## DSL Example (Final Vision)

```python
from charybdis import kernel, Tile, warpgroup_mma, reduce
import charybdis.dtypes as dt

@kernel(warpgroups=2)
def flash_attention(
    Q: Tile[dt.f16, (seqlen, d_head)],
    K: Tile[dt.f16, (seqlen, d_head)],
    V: Tile[dt.f16, (seqlen, d_head)],
    O: Tile[dt.f32, (seqlen, d_head)],
):
    scores = warpgroup_mma(Q, K.T)           # QK^T
    scores = scores - reduce(scores, "max")  # numerically stable
    scores = exp(scores)
    scores = scores / reduce(scores, "sum")  # softmax
    O[:] = warpgroup_mma(scores, V)          # weighted sum
```

---

## References

- [Triton](https://github.com/triton-lang/triton) — decorator-based GPU DSL, primary syntax inspiration. Uses `nvvm`/PTX lowering and nanobind for Python bindings.
- [TLX](https://github.com/facebookexperimental/triton/tree/tlx) — Triton extension for warpgroup-level ops, reference for warpgroup tiling semantics.
- [Mosaic GPU](https://github.com/jax-ml/jax/tree/main/jax/experimental/mosaic) — MLIR-based Hopper/Blackwell kernel DSL. Uses `nvgpu` dialect and upstream MLIR Python bindings. Primary compiler architecture reference for Charybdis.
