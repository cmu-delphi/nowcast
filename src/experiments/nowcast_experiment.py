"""
===============
=== Purpose ===
===============

Retrospective nowcasting for multiple experimental protocols, including:
  - ablation: removal of individual sensors
  - abscission: removal of geographic tiers
    - option 1: nowcast 1 season using 8+ sensors
    - option 2: nowcast 6+ seasons using 3 sensors
  - covariance: use different covariance estimation methods
  - vanilla: the operational nowcast, as a control and sanity check

The operational nowcast (the "vanilla" experiment) should be equivalent to:
  - the "covariance" experiment with parameter "bd2", corresponding to the
    covariance estimation method "BlendDiagonal2" in ../fusion/covariance.py
  - the "abscise1" experiment with parameter "state", which means to use all
    sensors at all locations (for the short period of time during which all
    8 sensors were simultaneously available)
"""

# standard library
import argparse
import csv

# first party
from delphi.epidata.client.delphi_epidata import Epidata
from delphi.nowcast.fusion import covariance
from delphi.nowcast.fusion.nowcast import Nowcast
from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.operations import secrets
from delphi.utils.epiweek import range_epiweeks
from delphi.utils.geo.locations import Locations


class InsufficientDataException(Exception):
  """An Exception indicating that there is not enough training data."""


class UnknownSensorException(Exception):
  """An Exception indicating that a sensor is unknown."""


class InvalidExperimentException(Exception):
  """An Exception indicating that an experiment is invalid."""


class NowcastExperiment:
  """Produces retrospective nowcasts of ILI in the US."""

  class Provider:
    """
    Provides instances for the experiment; can be overridden in unit tests.
    """

    def get_data_source(self, epidata, sensors, locations):
      """Return a FluDataSource instance."""
      return FluDataSource(epidata, sensors, locations)

    def get_nowcast(self, data_source, cov_impl):
      """Return a Nowcast instance."""
      return Nowcast(
          data_source,
          shrinkage=cov_impl,
          min_observations=NowcastExperiment.MIN_OBSERVATIONS)

    def call_with_file(self, filename, func):
      """Run the given function with a file object."""
      with open(filename, 'w', newline='') as file_obj:
        func(file_obj)

  # the number of weekly observations required to start making nowcasts
  MIN_OBSERVATIONS = 5

  @staticmethod
  def new_instance():
    """Return a production-ready instance."""
    return NowcastExperiment(
        NowcastExperiment.Provider(), Epidata, FluDataSource.new_instance())

  def __init__(self, provider, epidata, data_source):
    self.provider = provider
    self.epidata = epidata
    self.data_source = data_source

  @staticmethod
  def get_locations_at_resolution(resolution):
    """
    Return a list of locations in which sensor readings should be available.
    """

    # all experiments include (a copy of) the national tier
    locations = Locations.nat_list[::]

    if resolution in ('regional', 'state'):
      # all types of regions
      locations.extend(Locations.hhs_list)
      locations.extend(Locations.cen_list)

    if resolution == 'state':
      # state and below
      locations.extend(Locations.ny_state_list)
      locations.extend(Locations.atom_list)

    return locations

  def get_weeks_in_nowcast(self, sensor, min_observations):
    """Return a list of weeks on which nowcasts should include this sensor."""

    # get the range of possible nowcast weeks
    all_weeks = self.data_source.get_weeks()
    week_range = self.epidata.range(min(all_weeks), max(all_weeks))

    # get sensor data for US nationally on those weeks
    response = self.epidata.sensors(
        secrets.api.sensors, sensor, 'nat', week_range)
    rows = self.epidata.check(response)

    # extract the weeks from the returned data
    sensor_weeks = sorted(r['epiweek'] for r in rows)
    if len(sensor_weeks) <= min_observations:
      msg = 'sensor %s available <= %d weeks' % (sensor, min_observations)
      raise InsufficientDataException(msg)

    # return nowcast-able weeks based on this sensor
    return sensor_weeks[min_observations:]

  def get_ablation_parameters(self, ablate):
    """Return parameters for the 'ablation' experiment."""

    # all except user-specified sensor; all locations; weeks on which the
    # held-out sensor would be included in the nowcast; default covariance
    # method
    if ablate not in FluDataSource.SENSORS:
      raise UnknownSensorException('unknown sensor: %s' % ablate)
    sensors = [s for s in FluDataSource.SENSORS if s != ablate]
    locations = Locations.region_list
    weeks = self.get_weeks_in_nowcast(
        ablate, NowcastExperiment.MIN_OBSERVATIONS)
    cov_impl = covariance.BlendDiagonal2
    return sensors, locations, weeks, cov_impl

  def get_abscission1_parameters(self, abscise1):
    """Return parameters for the 'abscission1' experiment."""

    # all sensors; user-specified location tier; weeks on which all sensors
    # were available; default covariance method
    sensors = FluDataSource.SENSORS
    locations = NowcastExperiment.get_locations_at_resolution(abscise1)
    weeks = list(range_epiweeks(201445, 201520, inclusive=True))
    cov_impl = covariance.BlendDiagonal2
    return sensors, locations, weeks, cov_impl

  def get_abscission2_parameters(self, abscise2):
    """Return parameters for the 'abscission2' experiment."""

    # high-resolution sensors; user-specified location tier; weeks on which
    # sensors are all available; default covariance method
    sensors = ['twtr', 'cdc', 'sar3']
    locations = NowcastExperiment.get_locations_at_resolution(abscise2)
    latest_week = self.data_source.get_most_recent_issue()
    weeks = list(range_epiweeks(201330, latest_week, inclusive=True))
    cov_impl = covariance.BlendDiagonal2
    return sensors, locations, weeks, cov_impl

  def get_covariance_parameters(self, cov_name):
    """Return parameters for the 'covariance' experiment."""

    # all sensors; all locations; all weeks; user-specified covariance method
    # (see ../fusion/covariance.py)
    sensors = FluDataSource.SENSORS
    locations = Locations.region_list
    weeks = self.data_source.get_weeks()[NowcastExperiment.MIN_OBSERVATIONS:]
    cov_impl = {
      'bd0': covariance.BlendDiagonal0,
      'bd1': covariance.BlendDiagonal1,
      'bd2': covariance.BlendDiagonal2,
    }[cov_name]
    return sensors, locations, weeks, cov_impl

  def get_vanilla_parameters(self):
    """Return parameters for the 'vanilla' experiment."""

    # all sensors; all locations; all weeks; default covariance method
    sensors = FluDataSource.SENSORS
    locations = Locations.region_list
    weeks = self.data_source.get_weeks()[NowcastExperiment.MIN_OBSERVATIONS:]
    cov_impl = covariance.BlendDiagonal2
    return sensors, locations, weeks, cov_impl

  def get_values_for_experiment(
      self,
      ablate=None,
      abscise1=None,
      abscise2=None,
      covariance=None,
      vanilla=None):
    """Return parameters corresponding to the given experiment."""

    if ablate:
      return self.get_ablation_parameters(ablate)
    if abscise1:
      return self.get_abscission1_parameters(abscise1)
    if abscise2:
      return self.get_abscission2_parameters(abscise2)
    if covariance:
      return self.get_covariance_parameters(covariance)
    if vanilla:
      return self.get_vanilla_parameters()

  def save_to_file(self, file_obj, weeks, nowcasts):
    """Extract and save the results."""
    writer = csv.writer(file_obj)
    for week, nowcast in zip(weeks, nowcasts):
      for location, value, stdev in nowcast:
        writer.writerow([week, location, float(value), float(stdev)])

  def run_experiment(
      self, filename, ablate, abscise1, abscise2, covariance, vanilla):
    """
    Produce and save nowcasts as indicated by the given experiment parameters.
    """

    # get parameters for this experiment
    sensors, locations, weeks, cov_impl = self.get_values_for_experiment(
        ablate, abscise1, abscise2, covariance, vanilla)

    # prefetch bulk data
    data_source = self.provider.get_data_source(
        self.epidata, sensors, locations)
    data_source.prefetch(max(weeks))

    # compute the nowcasts
    nowcaster = self.provider.get_nowcast(data_source, cov_impl)
    nowcasts = nowcaster.batch_nowcast(weeks)

    # save the nowcasts
    print('saving nowcasts to %s' % filename)
    func = lambda file_obj: self.save_to_file(file_obj, weeks, nowcasts)
    self.provider.call_with_file(filename, func)


def get_argument_parser():
  """Define command line arguments and usage."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      'filename',
      help='output filename (*.csv)')
  parser.add_argument(
      '--ablate',
      choices=FluDataSource.SENSORS,
      help='ablation experiment, leaving out this sensor')
  parser.add_argument(
      '--abscise1',
      choices=('national', 'regional', 'state'),
      help='abscission experiment (all sensors), with this resolution')
  parser.add_argument(
      '--abscise2',
      choices=('national', 'regional', 'state'),
      help='abscission experiment (hi-res sensors), with this resolution')
  parser.add_argument(
      '--covariance',
      choices=('bd0', 'bd1', 'bd2'),
      help='covariance experiment, using this algorithm')
  parser.add_argument(
      '--vanilla',
      default=None,
      action='store_true',
      help='a control; unmodified operational nowcasting')
  return parser


def validate_args(args):
  """Validate and return command line arguments."""
  values = [
    args.ablate, args.abscise1, args.abscise2, args.covariance, args.vanilla
  ]
  if sum(v is not None for v in values) != 1:
    raise InvalidExperimentException('exactly one experiment must be run')
  return [args.filename] + values


def main(*args):
  """Run this script from the command line."""
  NowcastExperiment.new_instance().run_experiment(*args)


if __name__ == '__main__':
  main(*validate_args(get_argument_parser().parse_args()))
