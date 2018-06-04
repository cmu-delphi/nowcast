"""Unit tests for nowcast_experiment.py."""

# standard library
import unittest
from unittest.mock import MagicMock

# first party
from delphi.nowcast.fusion import covariance
from delphi.nowcast.fusion.nowcast import Nowcast
from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.utils.geo.locations import Locations

# py3tester coverage target
__test_target__ = 'delphi.nowcast.experiments.nowcast_experiment'


class UnitTests(unittest.TestCase):
  """Basic unit tests."""

  def setUp(self):
    self.mock_provider = MagicMock()
    self.mock_epidata = MagicMock()
    self.mock_data_source = MagicMock()
    self.experiment = NowcastExperiment(
        self.mock_provider, self.mock_epidata, self.mock_data_source)

  def test_get_argument_parser(self):
    """An ArgumentParser should be returned."""
    self.assertIsInstance(get_argument_parser(), argparse.ArgumentParser)

  def test_validate_args(self):
    """Arguments should be validated."""

    new_args = lambda: MagicMock(
        ablate=None,
        abscise1=None,
        abscise2=None,
        covariance=None,
        vanilla=None,
        filename='out.csv')

    with self.subTest(name='no experiment given'):
      args = new_args()
      with self.assertRaises(InvalidExperimentException):
        validate_args(args)

    with self.subTest(name='two experiments given'):
      args = new_args()
      args.ablate = 'gft'
      args.abscise1 = 'state'
      with self.assertRaises(InvalidExperimentException):
        validate_args(args)

    with self.subTest(name='valid example'):
      args = new_args()
      args.vanilla = True
      expected = ['out.csv', None, None, None, None, True]
      self.assertEqual(validate_args(args), expected)

  def test_nowcast_experiment_new_instance(self):
    """Create a NowcastExperiment instance with default parameters."""
    inst = NowcastExperiment.new_instance()
    self.assertIsInstance(inst, NowcastExperiment)

  def test_default_provider(self):
    """Provide default instances of required objects."""

    provider = NowcastExperiment.Provider()
    data_source = provider.get_data_source(None, None, None)
    nowcast = provider.get_nowcast(None, None)

    self.assertIsInstance(data_source, FluDataSource)
    self.assertIsInstance(nowcast, Nowcast)

  def test_get_locations_at_resolution(self):
    """Return locations needed for abscission experiments."""

    with self.subTest(name='national'):
      locations = NowcastExperiment.get_locations_at_resolution('national')
      self.assertEqual(locations, Locations.nat_list)

    with self.subTest(name='regional'):
      locations = NowcastExperiment.get_locations_at_resolution('regional')
      expected = Locations.nat_list + Locations.hhs_list + Locations.cen_list
      self.assertEqual(locations, expected)

    with self.subTest(name='state'):
      locations = NowcastExperiment.get_locations_at_resolution('state')
      self.assertEqual(locations, Locations.region_list)

  def test_get_weeks_in_nowcast(self):
    """Return weeks on which a sensor will be included in the nowcast."""

    self.mock_data_source.get_weeks.return_value = [2, 3, 4, 5, 6, 7]

    def set_fake_epidata(weeks):
      self.mock_epidata.check.return_value = [{'epiweek': w} for w in weeks]

    with self.subTest(name='enough historical data'):
      set_fake_epidata([3, 5, 7])
      result = self.experiment.get_weeks_in_nowcast('sar3', 2)
      self.assertEqual(result, [7])

    with self.subTest(name='not enough historical data'):
      set_fake_epidata([4, 6])
      with self.assertRaises(InsufficientDataException):
        self.experiment.get_weeks_in_nowcast('sar3', 2)

  def test_get_ablation_parameters_requires_valid_sensor(self):
    """Raise Exception for invalid sensor."""

    with self.assertRaises(UnknownSensorException):
      self.experiment.get_ablation_parameters('x')

  def test_get_ablation_parameters(self):
    """Return sensible ablation parameters."""

    week_list = list(range(201801, 201809))
    self.mock_data_source.get_weeks.return_value = week_list
    self.mock_epidata.check.return_value = [{'epiweek': w} for w in week_list]

    params = self.experiment.get_ablation_parameters('arch')
    sensors, locations, weeks, cov_impl = params

    actual_sensors = set(sensors)
    expected_sensors = set(FluDataSource.SENSORS) - {'arch'}
    self.assertEqual(actual_sensors, expected_sensors)
    self.assertEqual(locations, Locations.region_list)
    self.assertEqual(weeks, week_list[NowcastExperiment.MIN_OBSERVATIONS:])
    self.assertEqual(cov_impl, covariance.BlendDiagonal2)

  def test_get_abscission1_parameters(self):
    """Return sensible abscission1 parameters."""

    params = self.experiment.get_abscission1_parameters('national')
    sensors, locations, weeks, cov_impl = params

    self.assertEqual(sensors, FluDataSource.SENSORS)
    self.assertEqual(locations, Locations.nat_list)
    self.assertIn(201445, weeks)
    self.assertIn(201520, weeks)
    self.assertEqual(cov_impl, covariance.BlendDiagonal2)

  def test_get_abscission2_parameters(self):
    """Return sensible abscission2 parameters."""

    self.mock_data_source.get_most_recent_issue.return_value = 201820

    params = self.experiment.get_abscission2_parameters('national')
    sensors, locations, weeks, cov_impl = params

    self.assertEqual(set(sensors), set(['cdc', 'sar3', 'twtr']))
    self.assertEqual(locations, Locations.nat_list)
    self.assertIn(201330, weeks)
    self.assertIn(201820, weeks)
    self.assertEqual(cov_impl, covariance.BlendDiagonal2)

  def test_get_covariance_parameters(self):
    """Return sensible covariance parameters."""

    self.mock_data_source.get_weeks.return_value = [1, 2, 3]

    params = self.experiment.get_covariance_parameters('bd0')
    sensors, locations, weeks, cov_impl = params

    self.assertEqual(sensors, FluDataSource.SENSORS)
    self.assertEqual(locations, Locations.region_list)
    self.assertEqual(weeks, [1, 2, 3])
    self.assertEqual(cov_impl, covariance.BlendDiagonal0)

  def test_get_vanilla_parameters(self):
    """Return parameters used in operational nowcasting."""

    self.mock_data_source.get_weeks.return_value = [1, 2, 3]

    params = self.experiment.get_vanilla_parameters()
    sensors, locations, weeks, cov_impl = params

    self.assertEqual(sensors, FluDataSource.SENSORS)
    self.assertEqual(locations, Locations.region_list)
    self.assertEqual(weeks, [1, 2, 3])
    self.assertEqual(cov_impl, covariance.BlendDiagonal2)

  def test_get_values_for_experiment(self):
    """Return parameters for each experiment type."""

    week_list = list(range(201801, 201810))
    self.mock_data_source.get_most_recent_issue.return_value = max(week_list)
    self.mock_data_source.get_weeks.return_value = week_list
    self.mock_epidata.check.return_value = [{'epiweek': w} for w in week_list]

    with self.subTest(name='ablate'):
      params = self.experiment.get_values_for_experiment(ablate='gft')
      self.assertEqual(len(params), 4)
      self.assertNotIn('gft', params[0])
      self.assertIn('ght', params[0])

    with self.subTest(name='abscise1'):
      params = self.experiment.get_values_for_experiment(abscise1='regional')
      self.assertEqual(len(params), 4)
      self.assertNotIn('tx', params[1])
      self.assertIn('hhs6', params[1])

    with self.subTest(name='abscise2'):
      params = self.experiment.get_values_for_experiment(abscise2='regional')
      self.assertEqual(len(params), 4)
      self.assertNotIn('tx', params[1])
      self.assertIn('hhs6', params[1])

    with self.subTest(name='covariance'):
      params = self.experiment.get_values_for_experiment(covariance='bd1')
      self.assertEqual(len(params), 4)
      self.assertEqual(params[3], covariance.BlendDiagonal1)

    with self.subTest(name='vanilla'):
      params = self.experiment.get_values_for_experiment(vanilla=True)
      self.assertEqual(len(params), 4)
      self.assertEqual(params[0], FluDataSource.SENSORS)
      self.assertIn('tx', params[1])
      self.assertEqual(params[2][-1], 201809)
      self.assertEqual(params[3], covariance.BlendDiagonal2)

  def test_save_to_file(self):
    """Save the nowcast to a CSV file."""

    class FakeFile:

      def __init__(self):
        self.lines = []

      def write(self, d):
        self.lines.append(d.strip())

    file_obj = FakeFile()
    weeks = [50, 51, 52]
    nowcasts = [
      [('aa', 10, 20), ('ba', 30, 40)],
      [('ab', 11, 21), ('bb', 31, 41)],
      [('ac', 12, 22), ('bc', 32, 42)],
    ]

    self.experiment.save_to_file(file_obj, weeks, nowcasts)

    self.assertEqual(file_obj.lines, [
      '50,aa,10.0,20.0',
      '50,ba,30.0,40.0',
      '51,ab,11.0,21.0',
      '51,bb,31.0,41.0',
      '52,ac,12.0,22.0',
      '52,bc,32.0,42.0',
    ])

  def test_run_experiment(self):
    """Run an experiment end-to-end."""

    # vanilla
    args = ['filename'] + [None] * 4 + [True]
    self.mock_data_source.get_weeks.return_value = [1, 2, 3]
    self.experiment.run_experiment(*args)

    self.assertTrue(self.mock_data_source.get_weeks.called)

    self.assertTrue(self.mock_provider.get_data_source.called)
    args, kwargs = self.mock_provider.get_data_source.call_args
    self.assertEqual(len(args), 3)
    self.assertEqual(args[0], self.mock_epidata)
    self.assertIn('gft', args[1])
    self.assertIn('nv', args[2])
    data_source = self.mock_provider.get_data_source()

    self.assertTrue(self.mock_provider.get_nowcast.called)
    args, kwargs = self.mock_provider.get_nowcast.call_args
    self.assertEqual(len(args), 2)
    self.assertEqual(args[0], data_source)
    self.assertEqual(args[1], covariance.BlendDiagonal2)
    nowcaster = self.mock_provider.get_nowcast()

    self.assertTrue(self.mock_provider.call_with_file.called)
    args, kwargs = self.mock_provider.call_with_file.call_args
    self.assertEqual(len(args), 2)
    self.assertIsInstance(args[0], str)
    self.assertTrue(callable(args[1]))

    self.assertTrue(data_source.prefetch.called)
    args, kwargs = data_source.prefetch.call_args
    self.assertEqual(len(args), 1)
    self.assertEqual(args[0], 3)

    self.assertTrue(nowcaster.batch_nowcast.called)
    args, kwargs = nowcaster.batch_nowcast.call_args
    self.assertEqual(len(args), 1)
    self.assertEqual(args[0], [1, 2, 3])
