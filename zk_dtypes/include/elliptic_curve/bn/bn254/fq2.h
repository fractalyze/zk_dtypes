#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQ2_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQ2_H_

#include <stdint.h>

#include "zk_dtypes/include/elliptic_curve/bn/bn254/fq.h"
#include "zk_dtypes/include/field/extension_field.h"

namespace zk_dtypes::bn254 {

template <typename BaseField>
class Fq2BaseConfig {
 public:
  constexpr static uint32_t kDegreeOverBaseField = 2;
  constexpr static BaseField kNonResidue = -1;
};

class Fq2StdConfig : public Fq2BaseConfig<FqStd> {
 public:
  constexpr static bool kUseMontgomery = false;

  using StdConfig = Fq2StdConfig;

  using BaseField = FqStd;
  using BasePrimeField = FqStd;
};

class Fq2Config : public Fq2BaseConfig<Fq> {
 public:
  constexpr static bool kUseMontgomery = true;

  using StdConfig = Fq2StdConfig;

  using BaseField = Fq;
  using BasePrimeField = Fq;
};

using Fq2 = ExtensionField<Fq2Config>;
using Fq2Std = ExtensionField<Fq2StdConfig>;

}  // namespace zk_dtypes::bn254

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN254_FQ2_H_
