# CLAUDE.md

## Project Overview
zk_dtypes is a NumPy dtype extension library for zero-knowledge cryptography.
Provides prime fields, extension fields, binary fields, elliptic curve types,
and narrow integers — all registered as first-class NumPy dtypes.

## Current Focus
Q2: E2E proving p99 ≤ 7s on 16 GPUs (excluding verification).
Sprint: E2E correctness — 5 test blocks Phase 1-3 bug-free.
Out of scope: Multi-zkVM 2nd backend, community building, internal tooling, external talks.

## Commands
- Build: `bazel build //...`
- Test: `bazel test //...`

## Why Decisions
- NumPy extension over standalone lib: seamless JAX/NumPy interop without conversion overhead at every call boundary.
- C extensions for field arithmetic: Python-only implementation would be 100x slower for core field operations.
- Montgomery form as default: eliminates repeated conversion in proving pipelines that chain multiple field operations.

## Rules
- New field types MUST include property-based tests verifying all field axioms.
- Do NOT add dtypes without registering them in NumPy's type system.
- Do NOT break field axiom properties (associativity, commutativity, distributivity, identity, inverse).
- Montgomery `_mont` suffix and extension `x{degree}` suffix are mandatory naming conventions.
- Always run `bazel test //...` before committing.

## Invisible Traps
- `bn254_sf`/`bn254_sf_mont` are prime field (scalar field) types, NOT elliptic curve types. EC types are `bn254_g1_affine`, `bn254_g2_affine`, etc. Confusing them compiles but produces nonsense.
- Montgomery conversion test direction: `from_mont(to_mont(x)) == x` (start from standard element). C++ API uses `MontReduce()`.
- Narrow integers include int128/uint128/int256/uint256 — easily missed when enumerating types.

## Knowledge Files
Read ONLY when relevant to your current task:
@.claude/knowledge/architecture.md — Type hierarchy, C/Python binding layers
@.claude/knowledge/testing-guide.md — Property-based testing for field axioms
@.claude/knowledge/solutions.md — Past bug resolution patterns
