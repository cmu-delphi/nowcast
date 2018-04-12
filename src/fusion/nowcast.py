"""
===============
=== Purpose ===
===============

Generates (w)ILI nowcasts within the US through sensor fusion of digital
surveillance.
"""

# standard library
import abc
import itertools

# third party
import numpy as np

# first party
from delphi.nowcast.fusion import covariance
from delphi.nowcast.fusion import fusion
from delphi.nowcast.fusion.us_fusion import UsFusion
from delphi.utils.epiweek import split_epiweek


class DataSource(abc.ABC):
  """The interface by which all input data is provided."""

  @abc.abstractmethod
  def get_locations(self):
    """Return a list of possible locations."""

  @abc.abstractmethod
  def get_missing_locations(self, epiweek):
    """Return a tuple of locations which did not report on the given week."""

  @abc.abstractmethod
  def get_sensors(self):
    """Return a list of sensor names."""

  @abc.abstractmethod
  def get_weeks(self):
    """Return a list of weeks on which truth and sensors are both available."""

  @abc.abstractmethod
  def get_truth_value(self, epiweek, location):
    """Return ground truth (w)ILI."""

  @abc.abstractmethod
  def get_sensor_value(self, epiweek, location, name):
    """Return a sensor reading."""


class Nowcast:
  """Produces nowcasts with a given data source and shrinkage strategy."""

  """
  Creates a new Nowcast instance with the given configuration.

  input:
    data_source: a DataSource instance
    shrinkage (optional): a subclass of covariance.ShrinkageMethod
    min_observations (optional): the minimum number of observations required
      for any given (sensor, location) pair
  """
  def __init__(
      self,
      data_source,
      shrinkage=covariance.BlendDiagonal2,
      min_observations=5):
    self.data_source = data_source
    self.shrinkage = shrinkage
    self.min_observations = min_observations

  @staticmethod
  def get_season(epiweek):
    """Return the first year of the season that contains the given epiweek."""
    year, week = split_epiweek(epiweek)
    if week < 40:
      year -= 1
    return year

  @staticmethod
  def compute_nowcast(
      input_locations,
      noise,
      reading,
      shrinkage,
      season=None,
      exclude_locations=()):
    """
    Computes a nowcast via sensor fusion.

    inputs:
      input_locations: a list of locations, corresponding to columns of the
        matrix `noise` and the vector `reading`
      noise: matrix of past sensor noise (sensor readings minus truth)
      reading: vector of current sensor readings
      shrinkage: a subclass of covariance.ShrinkageMethod
      season (optional): the first year of the season that contains the epiweek
        being nowcasted; population weights are updated at the start of each
        flu season
      exclude_locations (optional): a tuple of atomic locations that should be
        excluded from statespace (i.e. because num_providers is known to be
        zero)

    outputs:
      - The nowcast for this week; a tuple of (location, (w)ILI, stdev) tuples.
    """

    # determine statespace and estimate covariance
    H, W, output_locations = UsFusion.determine_statespace(
        input_locations, season=season, exclude_locations=exclude_locations)
    R = covariance.mle_cov(noise, shrinkage)

    # apply the sensor fusion kernel
    x, P = fusion.fuse(reading, R, H)
    y, S = fusion.extract(x, P, W)

    # extract standard deviation vector from posterior covariance
    stdev = np.sqrt(np.diag(S))

    # return the nowcast for this week
    return tuple(zip(output_locations, y, stdev))

  def get_sensor_data_for_all_weeks(self, test_weeks):
    """
    Return all training and testing data for the given weeks.

    The main result is a pair of matrices representing training and testing
    data, respectively. For both of these, columns represent a particular
    sensor for a particular location, and rows represent observations on a
    particular epiweek.

    The two matrices will have the same columns, and the meaning of each column
    is indicated by the returned list of columns.

    input:
      test_weeks: a list of epiweeks for which nowcasts should be generated

    output:
      a tuple consisting of:
        - a list of (sensor, location) tuples corresponding to columns
        - a matrix of sensor noise (readings minus ground truth) for training
        - a matrix of sensor readings for testing
    """

    # get locations and sensors
    locations = self.data_source.get_locations()
    sensors = self.data_source.get_sensors()
    train_weeks = self.data_source.get_weeks()
    num_inputs = len(sensors) * len(locations)

    # exclude training weeks that are later than all testing weeks
    last_test_week = max(test_weeks)
    train_weeks = [week for week in train_weeks if week < last_test_week]

    # create empty (np.nan) matrices for sensor noise and readings
    sensor_noise = np.full((len(train_weeks), num_inputs), np.nan)
    sensor_readings = np.full((len(test_weeks), num_inputs), np.nan)

    # fill testing and training matrices
    inputs = list(itertools.product(sensors, locations))
    for col, (sen, loc) in enumerate(inputs):

      # training data
      for row, week in enumerate(train_weeks):
        sensor = self.data_source.get_sensor_value(week, loc, sen)
        truth = self.data_source.get_truth_value(week, loc)
        if sensor is not None and truth is not None:
          sensor_noise[row, col] = sensor - truth

      # testing data
      for row, week in enumerate(test_weeks):
        value = self.data_source.get_sensor_value(week, loc, sen)
        if value is not None:
          sensor_readings[row, col] = value

    # remove empty columns
    get_finite_columns = lambda data: np.any(np.isfinite(data), axis=0)
    noise_finite = get_finite_columns(sensor_noise)
    readings_finite = get_finite_columns(sensor_readings)
    keep_columns = np.logical_and(noise_finite, readings_finite)
    inputs = list(itertools.compress(inputs, keep_columns))
    sensor_noise = sensor_noise[:, keep_columns]
    sensor_readings = sensor_readings[:, keep_columns]

    # return column definitions and data
    return inputs, sensor_noise, sensor_readings

  def get_sensor_data_for_week(
      self, inputs, sensor_noise, week, week_reading, exclude_locations=()):
    """
    Return training data and sensor readings for the given week.

    This function is very similar to `get_sensor_data_for_all_weeks`, except
    here (1) empty rows and columns and (2) rows for future weeks relative to
    the given week are removed. In the returned matrices, there will be at
    least one non-empty entry in all rows and columns.

    Empty columns are removed from the list of inputs. This way, the returned
    list matches the columns of the returned matrices as before. This function
    only returns the list of locations, not (sensor, location) tuples.

    input:
      inputs: a list of (sensor, location) pairs representing matrix columns
      sensor_noise: matrix of training data (sensor readings minus truth)
      week: the epiweek for which the nowcast will be made
      week_reading: the vector of sensor readings for this week
      exclude_locations (optional): a tuple of atomic locations that should be
        excluded from statespace (i.e. because num_providers is known to be
        zero)

    output:
      a tuple consisting of:
        - a list of locations corresponding to columns
        - a matrix of sensor noise (readings minus ground truth) for training
        - a vector of sensor readings for nowcasting
    """

    # select training data in the past relative to this week
    train_weeks = self.data_source.get_weeks()[:sensor_noise.shape[0]]
    past_weeks = np.array(train_weeks) < week
    noise = sensor_noise[past_weeks, :]

    # select all rows with at least one observation
    noise_present = np.isfinite(noise)
    keep_rows = np.any(noise_present, axis=1)

    # select all columns with at least N observations
    noise_columns = np.sum(noise_present, axis=0) >= self.min_observations
    reading_columns = np.isfinite(week_reading)
    keep_columns = np.logical_and(noise_columns, reading_columns)

    # exclude columns corresponding to excluded locations
    keep_locs = [loc not in exclude_locations for (name, loc) in inputs]
    keep_columns = np.logical_and(keep_columns, keep_locs)

    # the list of locations corresponding to non-empty columns
    selected_inputs = list(itertools.compress(inputs, keep_columns))
    input_locations = tuple(loc for (name, loc) in selected_inputs)

    # remove rows and columns not selected above
    noise = noise[keep_rows, :][:, keep_columns]
    week_reading = week_reading[keep_columns]

    # return column definitions and data
    return input_locations, noise, week_reading

  def batch_nowcast(self, test_weeks):
    """
    Return a list of nowcasts, one for each test week.

    Note that the model is retrained for each test week, and training data is
    excluded for training weeks that are greater than or equal to the test
    week. In other words, this function excludes training data from the future,
    relative to the test data.

    Despite retraining, batched nowcasting (i.e. calling with multiple test
    weeks) is more efficient than repeated calls because some data structures
    are shared across iterations.

    input:
      test_weeks: a list of epiweeks for which nowcasts should be generated

    output:
      A list of nowcasts, one per test week. Each nowcast is a tuple of
      (location, (w)ILI, stdev) tuples.
    """

    # collect all training and testing data up-front
    inputs, noise, readings = self.get_sensor_data_for_all_weeks(test_weeks)

    # nowcast each week separately
    weekly_nowcasts = []
    for week, week_reading in zip(test_weeks, readings):

      # possibly exclude non-reporting locations (retrospective nowcasts only)
      exclude_locations = self.data_source.get_missing_locations(week)

      # get training and testing data "as of" the current week
      week_inputs, week_noise, week_reading = self.get_sensor_data_for_week(
          inputs, noise, week, week_reading, exclude_locations)

      # generate the nowcast in all possible locations for this week
      season = Nowcast.get_season(week)
      nowcast = Nowcast.compute_nowcast(
          week_inputs,
          week_noise,
          week_reading,
          self.shrinkage,
          season=season,
          exclude_locations=exclude_locations)
      weekly_nowcasts.append(nowcast)

      # show progress
      row = nowcast[0]
      args = (week, row[0], row[1], row[2])
      print('[%d] %s: %.3f (%.3f)' % args)

    # return the list of nowcasts
    return weekly_nowcasts
