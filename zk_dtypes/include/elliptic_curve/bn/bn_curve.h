/* Copyright 2026 The zk_dtypes Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_
#define ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_

#include <array>
#include <type_traits>

#include "zk_dtypes/include/control_flow_operation.h"
#include "zk_dtypes/include/elliptic_curve/pairing/g2_projective.h"
#include "zk_dtypes/include/elliptic_curve/pairing/pairing_friendly_curve.h"

namespace zk_dtypes {

// clang-format off
// BN curve pairing implementation (CRTP-enabled).
//
// Implements the optimal ate pairing for BN (Barreto-Naehrig) curves.
// The pairing e: G1 × G2 → GT is computed as:
//   e(P, Q) = FinalExponentiation(MillerLoop(P, Q))
//
// Template parameters:
//   Config  - Curve configuration (field types, BN parameter, ate loop count)
//   Derived - CRTP derived type. When void (default), uses concrete field
//             types from Config. When set (e.g., PairingCodeGen), enables
//             IR code generation via PairingTraits<Derived>.
//
// References:
// - "High-Speed Software Implementation of the Optimal Ate Pairing over
//   Barreto–Naehrig Curves" by Beuchat et al.
//   https://eprint.iacr.org/2010/354.pdf
// - "Faster hashing to G2" by Fuentes-Castañeda et al.
//   https://eprint.iacr.org/2011/297.pdf
// clang-format on
template <typename Config, typename Derived = void>
class BNCurve : public PairingFriendlyCurve<Config, Derived> {
 public:
  using Base = PairingFriendlyCurve<Config, Derived>;
  using Types = PairingTypes<Config, Derived>;
  using Fp = typename Types::Fp;
  using Fp12 = typename Base::Fp12;
  using Fp2 = typename Base::Fp2;
  using G1AffinePoint = typename Types::G1AffinePoint;
  using G2AffinePoint = typename Types::G2AffinePoint;
  using BoolType = typename Types::BoolType;

  // ==========================================================================
  // Fused multi-Miller loop: computes and consumes line coefficients in a
  // single pass, avoiding precomputation. Shared between
  // concrete and codegen modes via CRTP.
  //
  // For concrete mode (Derived = void):
  //   CFOp::For/If/Select use plain C++ loops/branches.
  // For codegen mode (Derived != void):
  //   CFOp::For emits scf.for, CFOp::If emits scf.if,
  //   CFOp::Select emits arith.select.
  // ==========================================================================

  // Miller loop iteration state: Fp12 accumulator + G2 projective points.
  template <int NumPairs>
  struct MillerState {
    Fp12 f;
    std::array<G2Projective<Config, Derived>, NumPairs> r;

    // Flatten state to Values for scf.for iter_args (codegen only).
    // These methods are only called in codegen mode; standard C++ template
    // instantiation ensures the bodies are only compiled when actually used.
    // SFINAE: only available in codegen mode (Derived != void).
    // VR is a deduced template parameter (mlir::ValueRange) to avoid
    // requiring the complete MLIR type in the declaration.
    template <typename D = Derived,
              std::enable_if_t<!std::is_void_v<D>, int> = 0>
    auto toValues() const {
      // V = the underlying value type (mlir::Value), dependent on D.
      using V = decltype(typename PairingTraits<D>::Fp().getValue());
      std::vector<V> v;
      v.reserve(1 + 3 * NumPairs);
      v.push_back(V(f));
      for (const auto& proj : r) {
        v.push_back(V(proj.x()));
        v.push_back(V(proj.y()));
        v.push_back(V(proj.z()));
      }
      return v;
    }

    template <typename VR, typename D = Derived,
              std::enable_if_t<!std::is_void_v<D>, int> = 0>
    static MillerState fromValues(VR vals) {
      MillerState s;
      s.f = Fp12(vals[0]);
      for (int p = 0; p < NumPairs; ++p) {
        s.r[p] = G2Projective<Config, Derived>(
            Fp2(vals[1 + 3 * p]), Fp2(vals[2 + 3 * p]), Fp2(vals[3 + 3 * p]));
      }
      return s;
    }

    // Component-wise select between two MillerStates.
    static MillerState Select(BoolType cond, const MillerState& a,
                              const MillerState& b) {
      MillerState s;
      using CFOp = ControlFlowOperation<BoolType>;
      s.f = CFOp::Select(cond, a.f, b.f);
      for (int p = 0; p < NumPairs; ++p) {
        s.r[p] = G2Projective<Config, Derived>::Select(cond, a.r[p], b.r[p]);
      }
      return s;
    }
  };

  // clang-format off
  // Fused multi-Miller loop.
  //
  // Computes f = ∏ᵢ f_{6x+2,Qᵢ}(Pᵢ) by computing and consuming line
  // coefficients in a single pass over the NAF bits.
  //
  // NumPairs is known at compile time (typically 2 or 4).
  // clang-format on
  template <int NumPairs>
  static Fp12 FusedMultiMillerLoop(
      const std::array<G1AffinePoint, NumPairs>& g1,
      const std::array<G2AffinePoint, NumPairs>& g2) {
    using CFOp = ControlFlowOperation<BoolType>;

    Fp two_inv = Fp::TwoInv();

    // Pre-negate G2 y-coordinates for NAF bit == -1 case.
    std::array<G2AffinePoint, NumPairs> neg_g2;
    for (int p = 0; p < NumPairs; ++p) {
      neg_g2[p] = -g2[p];
    }

    // Initialize: f = 1, R_p = (Q_p.x, Q_p.y, 1) for each pair.
    MillerState<NumPairs> init;
    init.f = Fp12::One();
    for (int p = 0; p < NumPairs; ++p) {
      init.r[p] =
          G2Projective<Config, Derived>(g2[p].x(), g2[p].y(), Fp2::One());
    }

    constexpr int64_t kNumIters =
        static_cast<int64_t>(std::size(Config::kAteLoopCount)) - 1;

    // Main fused Miller loop.
    auto result = CFOp::For(
        kNumIters, init,
        [&](auto iv, MillerState<NumPairs> state) -> MillerState<NumPairs> {
          // Step 1: f = f.Square()
          state.f = state.f.Square();

          // Step 2: Doubling step for each pair.
          for (int p = 0; p < NumPairs; ++p) {
            auto [newR, coeff] = state.r[p].Double(two_inv);
            state.r[p] = newR;
            Base::Ell(state.f, coeff, g1[p]);
          }

          // Step 3: Conditional addition (only when NAF bit != 0).
          auto naf_bit = GetNafBit(iv);
          auto naf_non_zero = IsNafNonZero(naf_bit);
          auto naf_positive = IsNafPositive(naf_bit);

          state = CFOp::If(
              naf_non_zero,
              [&]() -> MillerState<NumPairs> {
                auto s = state;
                for (int p = 0; p < NumPairs; ++p) {
                  // Select Q.y or -Q.y based on NAF sign.
                  auto addQ = G2AffinePoint(
                      g2[p].x(),
                      CFOp::Select(naf_positive, g2[p].y(), neg_g2[p].y()));
                  auto [newR, coeff] = s.r[p].Add(addQ);
                  s.r[p] = newR;
                  Base::Ell(s.f, coeff, g1[p]);
                }
                return s;
              },
              [&]() -> MillerState<NumPairs> { return state; });

          return state;
        });

    // Post-loop: BN-specific Frobenius corrections.
    for (int p = 0; p < NumPairs; ++p) {
      auto q1 = MulByCharacteristic(g2[p]);
      auto q2 = -MulByCharacteristic(q1);

      if constexpr (Config::kXIsNegative) {
        result.r[p] = result.r[p].Negate();
      }

      auto [r1, c1] = result.r[p].Add(q1);
      result.r[p] = r1;
      Base::Ell(result.f, c1, g1[p]);

      auto [r2, c2] = result.r[p].Add(q2);
      result.r[p] = r2;
      Base::Ell(result.f, c2, g1[p]);
    }

    return result.f;
  }

  // clang-format off
  // Final exponentiation: computes f^((q¹² - 1) / r)
  //
  // The exponent (q¹² - 1) / r factors as:
  //   (q¹² - 1) / r = (q⁶ - 1) · (q² + 1) · (q⁴ - q² + 1) / r
  //                   \______/   \______/   \_______________/
  //                easy part 1  easy part 2    hard part
  //
  // Easy parts are computed using Frobenius and conjugation.
  // Hard part uses the algorithm from Fuentes-Castañeda et al.
  // "Faster hashing to G2" (https://eprint.iacr.org/2011/297.pdf)
  // clang-format on
  static Fp12 FinalExponentiation(const Fp12& f) {
    // ==================== Easy Part ====================
    // Compute f^((q⁶ - 1) · (q² + 1))

    // Step 1: f^(q⁶ - 1)
    // For Fp12 = Fp6[w]/(w² - v), conjugation is f^(q⁶) = conj(f)
    Fp12 f1 = f.CyclotomicInverse();  // f^(q⁶) = conj(f)
    Fp12 f2 = f.Inverse();            // f⁻¹
    Fp12 r = f1 * f2;                 // f^(q⁶ - 1)

    // Step 2: raise to (q² + 1)
    f2 = r;
    r = r.template Frobenius<2>();  // r^(q²)
    r *= f2;                        // r^(q² + 1)

    // ==================== Hard Part ====================
    // Compute r^((q⁴ - q² + 1) / r) using Fuentes-Castañeda's algorithm.
    // This is optimized for BN curves where the exponent can be expressed
    // in terms of the BN parameter x.

    // Build intermediate powers of r
    Fp12 y0 = Base::PowByNegX(r);     // r^(-x)
    Fp12 y1 = y0.CyclotomicSquare();  // r^(-2x)
    Fp12 y2 = y1.CyclotomicSquare();  // r^(-4x)
    Fp12 y3 = y2 * y1;                // r^(-6x)
    Fp12 y4 = Base::PowByNegX(y3);    // r^(6x²)
    Fp12 y5 = y4.CyclotomicSquare();  // r^(12x²)
    Fp12 y6 = Base::PowByNegX(y5);    // r^(-12x³)

    // Adjust signs using cyclotomic inverse
    y3 = y3.CyclotomicInverse();  // r^(6x)
    y6 = y6.CyclotomicInverse();  // r^(12x³)

    // Combine powers to build the final result
    Fp12 y7 = y6 * y4;   // r^(12x³ + 6x²)
    Fp12 y8 = y7 * y3;   // r^(12x³ + 6x² + 6x)
    Fp12 y9 = y8 * y1;   // r^(12x³ + 6x² + 4x)
    Fp12 y10 = y8 * y4;  // r^(12x³ + 12x² + 6x)
    Fp12 y11 = y10 * r;  // r^(12x³ + 12x² + 6x + 1)

    // Apply Frobenius maps and combine
    Fp12 y12 = y9.template Frobenius<1>();  // y9^q
    Fp12 y13 = y12 * y11;
    Fp12 y8_frob = y8.template Frobenius<2>();  // y8^(q²)
    Fp12 y14 = y8_frob * y13;

    r = r.CyclotomicInverse();                    // r^(-1)
    Fp12 y15 = (r * y9).template Frobenius<3>();  // (r^(-1) · y9)^(q³)
    Fp12 result = y15 * y14;

    return result;
  }

 private:
  // Applies the Frobenius endomorphism and scales by twist constants.
  // Computes π(P) where π is the q-power Frobenius map, adjusted for
  // the twist isomorphism between E'(Fp2) and E(Fp12).
  static G2AffinePoint MulByCharacteristic(const G2AffinePoint& r) {
    Fp2 x = r.x().template Frobenius<1>() * TwistMulByQX();
    Fp2 y = r.y().template Frobenius<1>() * TwistMulByQY();
    return {x, y};
  }

  // Access twist constants. When Derived = void, these come from Config.
  // When Derived is provided, PairingTraits may supply them differently.
  static Fp2 TwistMulByQX() {
    if constexpr (std::is_void_v<Derived>) {
      return Config::kTwistMulByQX;
    } else {
      return PairingTraits<Derived>::TwistMulByQX();
    }
  }

  static Fp2 TwistMulByQY() {
    if constexpr (std::is_void_v<Derived>) {
      return Config::kTwistMulByQY;
    } else {
      return PairingTraits<Derived>::TwistMulByQY();
    }
  }

  // NAF bit access dispatchers.
  // In concrete mode (Derived = void), GetNafBit/IsNafNonZero/IsNafPositive
  // use plain C++ array access and comparisons.
  // In codegen mode (Derived != void), they delegate to PairingTraits
  // which emit MLIR ops (tensor.extract, arith.cmpi).

  // Concrete mode: index into constexpr array.
  template <typename D = Derived>
  static auto GetNafBit(std::enable_if_t<std::is_void_v<D>, int64_t> iv) {
    constexpr int64_t kSize =
        static_cast<int64_t>(std::size(Config::kAteLoopCount));
    return Config::kAteLoopCount[kSize - 2 - iv];
  }

  template <typename D = Derived>
  static auto IsNafNonZero(
      std::enable_if_t<std::is_void_v<D>, int8_t> naf_bit) {
    return naf_bit != 0;
  }

  template <typename D = Derived>
  static auto IsNafPositive(
      std::enable_if_t<std::is_void_v<D>, int8_t> naf_bit) {
    return naf_bit > 0;
  }

  // Codegen mode: delegate to PairingTraits.
  // Uses a generic type T so the declaration doesn't reference MLIR types
  // in concrete mode. SFINAE ensures only the correct overload is available.
  template <typename T, typename D = Derived,
            std::enable_if_t<!std::is_void_v<D>, int> = 0>
  static auto GetNafBit(T iv) {
    return PairingTraits<Derived>::GetNafBit(iv);
  }

  template <typename T, typename D = Derived,
            std::enable_if_t<!std::is_void_v<D>, int> = 0>
  static auto IsNafNonZero(T naf_bit) {
    return PairingTraits<Derived>::IsNafNonZero(naf_bit);
  }

  template <typename T, typename D = Derived,
            std::enable_if_t<!std::is_void_v<D>, int> = 0>
  static auto IsNafPositive(T naf_bit) {
    return PairingTraits<Derived>::IsNafPositive(naf_bit);
  }
};

}  // namespace zk_dtypes

#endif  // ZK_DTYPES_INCLUDE_ELLIPTIC_CURVE_BN_BN_CURVE_H_
