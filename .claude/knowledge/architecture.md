# zk_dtypes Architecture

## Type Hierarchy

```
NumPy dtype extensions
├── Narrow Integers: int2, int4, uint2, uint4, int128, uint128, int256, uint256
├── Prime Fields
│   ├── BabyBear (p=2013265921)
│   ├── Goldilocks (p=2^64-2^32+1)
│   ├── KoalaBear (p=2^31-2^24+1=2130706433)
│   ├── Mersenne31 (p=2^31-1)
│   ├── BN254 scalar field: bn254_sf, bn254_sf_mont
│   ├── BN254 Fq: bn254_fq, bn254_fq_mont
│   └── BN254 Fr: bn254_fr, bn254_fr_mont
├── Extension Fields
│   ├── BabyBearx4 (degree-4 extension)
│   ├── Goldilocksx3 (degree-3 extension)
│   └── KoalaBearx4
├── Binary Fields: binary_field_t0..t7
└── Elliptic Curves
    ├── BN254 G1: bn254_g1_affine, bn254_g1_jacobian, bn254_g1_xyzz (+ _mont variants)
    └── BN254 G2: bn254_g2_affine, bn254_g2_jacobian, bn254_g2_xyzz (+ _mont variants)
```

## Implementation Layers
1. **C/C++ core**: Field arithmetic implementations, NumPy dtype registration
2. **Python bindings**: NumPy-compatible interface, operator overloading
3. **Testing**: Property-based tests verifying field axioms

## Key Design Decisions
- Each dtype has both standard and Montgomery form (`_mont` suffix)
- All field ops must satisfy: associativity, commutativity, distributivity
- Extension field degree is encoded in the type name
