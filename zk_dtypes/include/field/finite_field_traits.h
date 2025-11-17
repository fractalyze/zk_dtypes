#ifndef ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_
#define ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_

namespace zk_dtypes {

template <typename Config, typename SFINAE = void>
class PrimeField;

template <typename Config>
class ExtensionField;

template <typename T>
class FiniteFieldTraits;

template <typename Config>
class FiniteFieldTraits<PrimeField<Config>> {
 public:
  using BasePrimeField = PrimeField<Config>;
};

template <typename Config>
class FiniteFieldTraits<ExtensionField<Config>> {
 public:
  using BasePrimeField = typename Config::BasePrimeField;
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_FIELD_FINITE_FIELD_TRAITS_H_
