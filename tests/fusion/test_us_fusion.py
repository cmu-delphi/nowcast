"""Unit tests for us_fusion.py."""

# standard library
from fractions import Fraction
import unittest

# first party
from delphi.utils.geo.locations import Locations

# py3tester coverage target
__test_target__ = 'delphi.nowcast.fusion.us_fusion'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def test_get_weight_row(self):
    # single atom
    weights = UsFusion.get_weight_row('ca', None, Locations.atom_list)
    self.assertEqual(sum(weights), 1)
    for w in weights:
      self.assertTrue(isinstance(w, Fraction))
      self.assertTrue(w in (0, 1))

    # some atoms
    weights = UsFusion.get_weight_row('hhs1', None, Locations.atom_list)
    self.assertEqual(sum(weights), 1)
    for w in weights:
      self.assertTrue(isinstance(w, Fraction))
      self.assertTrue(0 <= w < 1)

    # all atoms
    weights = UsFusion.get_weight_row('nat', None, Locations.atom_list)
    self.assertEqual(sum(weights), 1)
    for w in weights:
      self.assertTrue(isinstance(w, Fraction))
      self.assertTrue(0 < w < 1)

  def test_get_weight_matrix(self):
    # typical usage
    regions = Locations.region_list
    atoms = Locations.atom_list
    W = UsFusion.get_weight_matrix(regions, None, atoms).astype(np.float)
    self.assertEqual(W.shape, (len(regions), len(atoms)))
    self.assertTrue(np.allclose(np.sum(W, axis=1), 1))

    # single atom, matching
    W = UsFusion.get_weight_matrix(['pa'], None, ['pa']).astype(np.float)
    self.assertEqual(W.shape, (1, 1))
    self.assertTrue(np.isclose(W[0, 0], 1))

    # single atom, non-matching
    with self.assertRaises(Exception):
      UsFusion.get_weight_matrix(['pa'], None, ['ga']).astype(np.float)

  def test_determine_statespace(self):
    # typical invocation, uncached
    inputs = tuple(Locations.region_list * 3)
    result1 = UsFusion.determine_statespace(tuple(inputs))
    H, W, outputs = result1
    self.assertEqual(outputs, Locations.region_list)
    self.assertEqual(H.shape[0], len(inputs))
    self.assertEqual(W.shape[0], len(outputs))
    self.assertEqual(H.shape[1], W.shape[1])

    # typical invocation, cached
    result2 = UsFusion.determine_statespace(inputs[::1])
    self.assertIs(result1, result2)

    # typical invocation, uncached
    result3 = UsFusion.determine_statespace(inputs[::-1])
    self.assertIsNot(result1, result3)

    # retrospective invocation, uncached
    result4 = UsFusion.determine_statespace(inputs, season=2016)
    self.assertIsNot(result1, result4)

    # directly exclude an input location
    with self.assertRaises(Exception):
      inputs = tuple(Locations.atom_list)
      excludes = ('ar',)
      UsFusion.determine_statespace(inputs, exclude_locations=excludes)

    # indirectly exclude an input location
    with self.assertRaises(Exception):
      excludes = tuple(Locations.region_map['hhs2'])
      UsFusion.determine_statespace(('hhs2',), exclude_locations=excludes)

    # some locations aren't available for all seasons
    UsFusion.determine_statespace(('pr',), season=2013)
    with self.assertRaises(Exception):
      UsFusion.determine_statespace(('pr',), season=2012)

    # fusion of national and regional locations only
    inputs = Locations.nat_list + Locations.hhs_list + Locations.cen_list
    H, W, outputs = UsFusion.determine_statespace(tuple(inputs))
    self.assertTrue(set(outputs) > set(inputs))
    self.assertTrue(set(outputs) & Locations.atom_map.keys())
    self.assertEqual(H.shape[0], len(inputs))
    self.assertEqual(W.shape[0], len(outputs))
    self.assertEqual(H.shape[1], W.shape[1])
    self.assertTrue(H.shape[1] < len(inputs))
    self.assertIn('pa', outputs)
    self.assertNotIn('tx', outputs)

    # hhs2, all atoms
    inputs = ['nj', 'ny', 'jfk', 'pr', 'vi']
    expected_outputs = inputs + ['hhs2', 'ny_state']
    H, W, outputs = UsFusion.determine_statespace(tuple(inputs))
    self.assertEqual(set(outputs), set(expected_outputs))
    self.assertEqual(H.shape[1], 5)

    # hhs2, missing territories
    inputs = ['hhs2', 'nj', 'ny', 'jfk']
    expected_outputs = inputs + ['ny_state']
    H, W, outputs = UsFusion.determine_statespace(tuple(inputs))
    self.assertEqual(set(outputs), set(expected_outputs))
    self.assertEqual(H.shape[1], 4)

    # hhs2, missing New York atoms
    inputs = ['nj', 'ny_state', 'pr', 'vi']
    expected_outputs = inputs + ['hhs2']
    H, W, outputs = UsFusion.determine_statespace(tuple(inputs))
    self.assertEqual(set(outputs), set(expected_outputs))
    self.assertEqual(H.shape[1], 4)

    # hhs2, missing all of New York
    inputs = ['hhs2', 'nj', 'pr', 'vi']
    expected_outputs = inputs + ['ny_state']
    H, W, outputs = UsFusion.determine_statespace(tuple(inputs))
    self.assertEqual(set(outputs), set(expected_outputs))
    self.assertEqual(H.shape[1], 4)
