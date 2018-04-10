"""
===============
=== Purpose ===
===============

A wrapper for the Epidata API as used for nowcasting. Caching is used
extensively to reduce the number of requests made to the API.
"""


# standard library
import functools

# first party
from delphi.epidata.client.delphi_epidata import Epidata
from delphi.nowcast.fusion.nowcast import DataSource
from delphi.operations import secrets
from delphi.utils.epidate import EpiDate
from delphi.utils.epiweek import add_epiweeks, range_epiweeks
from delphi.utils.geo.locations import Locations


class FluDataSource(DataSource):
  """The interface by which all input data is provided."""

  # the first epiweek for which we have ground truth ILI in all locations
  FIRST_DATA_EPIWEEK = 201040

  def __init__(self, epidata, sensors):
    self.epidata = epidata
    self.sensors = sensors
    # cache for prefetching bulk flu data
    self.cache = {}

  @functools.lru_cache(maxsize=1)
  def get_locations(self):
    """Return a list of possible locations."""
    return Locations.region_list

  @functools.lru_cache(maxsize=None)
  def get_missing_locations(self, epiweek):
    """Return a tuple of locations which did not report on the given week."""
    all_locations = self.get_locations()
    available_locations = []
    for loc in all_locations:
      if self.get_truth_value(epiweek, loc) is not None:
        available_locations.append(loc)
    return tuple(set(all_locations) - set(available_locations))

  @functools.lru_cache(maxsize=1)
  def get_sensors(self):
    """Return a list of sensor names."""
    return self.sensors

  @functools.lru_cache(maxsize=1)
  def get_weeks(self):
    """Return a list of weeks on which truth and sensors are both available."""
    latest_week = self.get_most_recent_issue()
    week_range = range_epiweeks(
        FluDataSource.FIRST_DATA_EPIWEEK, latest_week, inclusive=True)
    return list(week_range)

  def get_truth_value(self, epiweek, location):
    """Return ground truth (w)ILI."""

    try:
      return self.cache['ilinet'][location][epiweek]
    except KeyError:
      print('cache miss: get_truth_value', epiweek, location)
      response = self.epidata.fluview(location, epiweek)
      if response['result'] != 1:
        return self.add_to_cache('ilinet', location, epiweek, None)
      data = response['epidata'][0]
      if data['num_providers'] == 0:
        return self.add_to_cache('ilinet', location, epiweek, None)
      return self.add_to_cache('ilinet', location, epiweek, data['wili'])

  @functools.lru_cache(maxsize=None)
  def get_sensor_value(self, epiweek, location, name):
    """Return a sensor reading."""

    try:
      return self.cache[name][location][epiweek]
    except KeyError:
      print('cache miss: get_sensor_value', epiweek, location, name)
      response = self.epidata.sensors(
          secrets.api.sensors, name, location, epiweek)
      if response['result'] != 1:
        return self.add_to_cache(name, location, epiweek, None)
      value = response['epidata'][0]['value']
      return self.add_to_cache(name, location, epiweek, value)

  @functools.lru_cache(maxsize=1)
  def get_most_recent_issue(self):
    """Return the most recent epiweek for which FluView data is available."""
    ew2 = EpiDate.today().get_ew()
    ew1 = add_epiweeks(ew2, -9)
    response = self.epidata.fluview('nat', self.epidata.range(ew1, ew2))
    issues = [row['issue'] for row in self.epidata.check(response)]
    return max(issues)

  def add_to_cache(self, name, location, epiweek, value):
    """Add the given value to the cache."""
    if name not in self.cache:
      self.cache[name] = {}
    if location not in self.cache[name]:
      self.cache[name][location] = {}
    self.cache[name][location][epiweek] = value
    return value

  def prefetch(self, epiweek):
    """
    Fetch all data in all locations up to the given epiweek.

    Requests are batched. This is significantly more efficient (and faster)
    than querying each sensor/location/epiweek data point individually.
    """

    def extract(response):
      if response['result'] == -2:
        return []
      return self.epidata.check(response)

    weeks = Epidata.range(FluDataSource.FIRST_DATA_EPIWEEK, epiweek)

    # loop over locations to avoid hitting the limit of ~3.5k rows
    for loc in self.get_locations():
      print('fetching %s...' % loc)

      # default to None to prevent cache misses on missing values
      for week in range_epiweeks(
          FluDataSource.FIRST_DATA_EPIWEEK, epiweek, inclusive=True):
        for name in ['ilinet'] + self.get_sensors():
          self.add_to_cache(name, loc, week, None)

      # ground truth
      response = self.epidata.fluview(loc, weeks, auth=secrets.api.fluview)
      for row in extract(response):
        # skip locations with no reporters
        if row['num_providers'] > 0:
          self.add_to_cache('ilinet', loc, row['epiweek'], row['wili'])

      # sensors readings
      for sen in self.get_sensors():
        response = self.epidata.sensors(secrets.api.sensors, sen, loc, weeks)
        for row in extract(response):
          self.add_to_cache(sen, loc, row['epiweek'], row['value'])
