# zk_dtypes

## Purpose

zk_dtypes is a C++ header-only library of zero-knowledge-friendly data types:
prime fields, extension fields, elliptic curves, and big integers. It provides
both standard and Montgomery representations with compile-time field
configuration. Downstream consumers include **prime-ir** (MLIR constant folding
and code generation) and **riscv-witness** (SP1 precompile trace computation).

## Key Design Patterns

### CRTP Operation Classes

Point arithmetic is implemented via CRTP: `AffinePoint<Curve>` inherits
`AffinePointOperation<AffinePoint<Curve>>`. The operation class calls back into
the derived class for `ToCoords()`, `FromCoords()`, `CreateJacobianPoint()`,
`GetCFOperation()`, etc.

This same CRTP mechanism is used by **prime-ir** `PointCodeGenBase` and
`PointOperationBase` — they inherit the same zk_dtypes operation classes,
providing MLIR-Value-based or FieldOperation-based coordinates instead of field
element values. This is why operation classes must use `GetCFOperation()` for
control flow rather than plain `if/else`.

### SFINAE Curve Family Dispatch

`AffinePoint<Curve>` is partially specialized via SFINAE on
`Curve::kType == CurveType::kShortWeierstrass` or `kTwistedEdwards`. This lets
both curve families share the `AffinePoint<>` name while having completely
different operation implementations. `PointTraits<>` is similarly SFINAE-split.

### Montgomery Representations

Every curve has both standard (`Config`) and Montgomery (`MontConfig`) variants.
The Mont config stores curve constants (a, b/d, Gx, Gy) in Montgomery form via
`BaseField::FromUnchecked(...)`. The standard config stores raw integer values.

**Pitfall**: Use constructors (not `FromUnchecked`) for value→field conversion.
Use `MontReduce()` for Montgomery→standard extraction. See knowledge-graph
pitfall `montgomery-integer-field-conversion-bug.md`.

### Full-Width Modulus Carry Bug

For fields where `kModulusBits == kStorageBits` (no spare bit in top limb),
`SlowMontMul` must propagate the final carry to `Reduce()`. secp256k1 Fq is the
canonical example. Curve25519 Fq has a spare bit (255 < 256) so it uses
`FastMontMul`.

## Directory & Naming Conventions

### Curve Directory Layout

Follows the `<family>/<instance>/` nesting when the family has multiple curves
sharing a base field:

| Pattern                       | Example                   | When                                  |
| ----------------------------- | ------------------------- | ------------------------------------- |
| `<curve>/g1.h`                | `secp256k1/g1.h`          | Standalone curve, one group           |
| `<family>/<curve>/g1.h`       | `bn/bn254/g1.h`           | Family with shared field              |
| `<family>/<curve>/g1.h, g2.h` | `bn/bn254/g1.h, g2.h`     | Pairing curve                         |
| `<field-family>/<curve>/g1.h` | `curve25519/ed25519/g1.h` | Different curve forms sharing a field |

Field files (`fq.h`, `fr.h`) live at the family level when shared.

### Namespace Convention

Namespaces mirror the instance, NOT the directory:

- `zk_dtypes::secp256k1` (flat — no family)
- `zk_dtypes::bn254` (instance, not `bn::bn254`)
- `zk_dtypes::ed25519` (instance, not `curve25519::ed25519`)

Field types from the family namespace are imported via `using`:

```cpp
namespace zk_dtypes::ed25519 {
using curve25519::Fq;
using curve25519::FqMont;
}
```

### Config Class Naming

| Curve Type | Config            | Mont Config           | Std Types                                     | Mont Types                         |
| ---------- | ----------------- | --------------------- | --------------------------------------------- | ---------------------------------- |
| SW         | `G1SwCurveConfig` | `G1SwCurveMontConfig` | `G1Curve`, `G1AffinePoint`                    | `G1CurveMont`, `G1AffinePointMont` |
| TE         | `G1TeCurveConfig` | `G1TeCurveMontConfig` | `G1Curve`, `G1AffinePoint`, `G1ExtendedPoint` | `G1CurveMont`, etc.                |

### BUILD Target Naming

- Field targets: `<family>_fq`, `<family>_fr` (e.g. `curve25519_fq`)
- Curve targets: `<instance>_g1` (e.g. `ed25519_g1`, `secp256k1_g1`)
- SW infra: `sw_curve`, `sw_affine_point`, `sw_jacobian_point`, `sw_point_xyzz`
- TE infra: `te_curve`, `te_affine_point`, `te_extended_point`

### Test Convention

- Small-prime test config in `<curve_type>/test/<type>_curve_config.h` (e.g.
  `twisted_edwards/test/te_curve_config.h` with F₁₃ curve)
- Affine/extended/jacobian point unittests in `tests/elliptic_curve/`
- Real-curve sanity tests (on-curve check, identity, inverse, double, Mont
  consistency) in the same directory

## Adding a New Curve

1. **Field configs** (`fq.h`, `fr.h`): Compute Montgomery constants
   (`kRSquared`, `kNPrime`, `kOne`) via Python. Verify
   `kNPrime * p ≡ -1 mod 2⁶⁴`.
1. **Curve config** (`g1.h`): Define `G1<Sw|Te>CurveConfig` with `kA`/`kB`/`kD`,
   generator coords, and Mont variant with `FromUnchecked`. Verify generator is
   on curve in a unittest.
1. **BUILD.bazel**: Add `cc_library` targets. Add to `all_types` deps and the
   appropriate type list macros in `all_types.h`.
1. **Tests**: On-curve check, identity addition, inverse, double=add(self),
   Mont/non-Mont consistency.

## Downstream Integration

### prime-ir

prime-ir uses zk_dtypes for:

- **Constant folding**: `PointOperationBase<Kind>::fromZkDtype()` converts
  zk_dtypes points to MLIR attributes, then folds via the same CRTP operation
  classes
- **Code generation**: `PointCodeGenBase<Kind>` inherits the operation classes
  with `FieldCodeGen` (MLIR Values) as the base field type
- **Curve recognition**: `KnownCurves.cpp` matches `ShortWeierstrassAttr` /
  `TwistedEdwardsAttr` parameters against zk_dtypes curve configs

When adding a new curve to zk_dtypes, prime-ir needs:

- `getPointType<T>()` case in `PointOperation.h`
- `getTwistedEdwardsAttr<T>()` or `getShortWeierstrassAttr<T>()` specialization
- BUILD dep on the new `@zk_dtypes` target

### riscv-witness

riscv-witness uses zk_dtypes for:

- **Precompile computation**: `CurveField<Tag>::G1Point` and `FqMont` in
  `HandleEcAdd<Tag>()` — reads raw limbs from emulated memory, converts to Mont,
  computes, converts back
- **Trace input generation**: `BatchEcAdd` computes intermediates needed by the
  MLIR filler

When adding a new curve, riscv-witness needs:

- `CurveField<NewTag>` specialization in `field_adapter.h`
- Syscall constant + dispatch case in `sp1_provider.cpp`
- Collector in `SP1PrecompileTraceContext`
- Chip registry entry + Rust trace tooling
