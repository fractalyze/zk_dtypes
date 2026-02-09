# Copyright 2026 The zk_dtypes Authors.
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

"""Tests for BN254 pairing_check."""

from absl.testing import absltest

import numpy as np
from zk_dtypes import bn254_g1_affine, bn254_g2_affine, pairing_check


class PairingCheckTest(absltest.TestCase):

  def test_import(self):
    """pairing_check should be callable."""
    self.assertTrue(callable(pairing_check))

  def test_generator_self_pairing(self):
    """e(G1, G2) · e(-G1, G2) == 1 should hold."""
    # Create G1 generator (scalar = 1)
    g1 = np.array(bn254_g1_affine(1))
    neg_g1 = np.array(-bn254_g1_affine(1))
    g1_points = np.array([g1, neg_g1])

    # Create G2 generator (scalar = 1)
    g2 = np.array(bn254_g2_affine(1))
    g2_points = np.array([g2, g2])

    result = pairing_check(g1_points, g2_points)
    self.assertTrue(result)

  def test_mismatched_pairing_fails(self):
    """e(G1, G2) · e(2·G1, G2) != 1 for non-trivial points."""
    g1 = np.array(bn254_g1_affine(1))
    g1_2 = np.array(bn254_g1_affine(2))
    g1_points = np.array([g1, g1_2])

    g2 = np.array(bn254_g2_affine(1))
    g2_points = np.array([g2, g2])

    result = pairing_check(g1_points, g2_points)
    self.assertFalse(result)

  def test_length_mismatch_raises(self):
    """Should raise ValueError when array lengths don't match."""
    g1_points = np.array([bn254_g1_affine(1)])
    g2_points = np.array([bn254_g2_affine(1), bn254_g2_affine(1)])

    with self.assertRaises(ValueError):
      pairing_check(g1_points, g2_points)

  def test_empty_raises(self):
    """Should raise ValueError for empty arrays."""
    g1_points = np.array([], dtype=bn254_g1_affine)
    g2_points = np.array([], dtype=bn254_g2_affine)

    with self.assertRaises(ValueError):
      pairing_check(g1_points, g2_points)

  def test_bilinearity_g1(self):
    """e(a·G1, G2) · e(b·G1, G2) · e(-(a+b)·G1, G2) == 1."""
    a, b = 3, 5

    g1_a = np.array(bn254_g1_affine(a))
    g1_b = np.array(bn254_g1_affine(b))
    neg_g1_sum = np.array(-bn254_g1_affine(a + b))
    g1_points = np.array([g1_a, g1_b, neg_g1_sum])

    g2 = np.array(bn254_g2_affine(1))
    g2_points = np.array([g2, g2, g2])

    result = pairing_check(g1_points, g2_points)
    self.assertTrue(result)

  def test_bilinearity_g2(self):
    """e(G1, a·G2) · e(-a·G1, G2) == 1 (requires G2 scalar mul)."""
    a = 5

    g1 = np.array(bn254_g1_affine(1))
    neg_g1_a = np.array(-bn254_g1_affine(a))
    g1_points = np.array([g1, neg_g1_a])

    g2_a = np.array(bn254_g2_affine(a))
    g2 = np.array(bn254_g2_affine(1))
    g2_points = np.array([g2_a, g2])

    result = pairing_check(g1_points, g2_points)
    self.assertTrue(result)


if __name__ == "__main__":
  absltest.main()
