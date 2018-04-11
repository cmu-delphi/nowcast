"""
===============
=== Purpose ===
===============

Nowcasting Influenza-like Illness via sensor fusion of digital surveillance!
"""

# standard library
import argparse

# first party
from delphi.epidata.client.delphi_epidata import Epidata
from delphi.nowcast.fusion.nowcast import Nowcast
from delphi.nowcast.util.database import NowcastDatabase
from delphi.nowcast.util.flu_data_source import FluDataSource
from delphi.utils.epiweek import add_epiweeks, range_epiweeks


class NowcastUpdate:
  """
  Produces both real-time and retrospective nowcasts of ILI in the US. Nowcasts
  are stored in the Delphi database and are accessible via the Epidata API.
  """

  # all known sensors, past and present
  SENSORS = ['gft', 'ght', 'twtr', 'wiki', 'cdc', 'epic', 'sar3', 'arch']

  @staticmethod
  def new_instance(test_mode):
    """
    Return a new instance under the default configuration. If `test_mode` is
    true, database changes will not be committed.
    """
    database = NowcastDatabase.new_instance(test_mode)
    data_source = FluDataSource(Epidata, NowcastUpdate.SENSORS)
    return NowcastUpdate(database, data_source)

  def __init__(self, database, data_source):
    self.database = database
    self.data_source = data_source

  def get_update_range(self, first_week, last_week):
    """Return the range of epiweeks to update."""

    # default to most recent issue if a week range isn't given
    if not last_week:
      # repeat previous nowcast in case new data is available
      first_week = self.data_source.get_most_recent_issue()
      # nowcast the first week without ilinet data
      last_week = add_epiweeks(first_week, 1)
    return first_week, last_week

  def update(self, first_week, last_week):
    """Nowcast the given range of weeks and save the result to the database."""

    # update the week range if needed
    first_week, last_week = self.get_update_range(first_week, last_week)
    print('nowcasting %d--%d' % (first_week, last_week))

    # prefetch bulk data
    self.data_source.prefetch(last_week)

    # compute the nowcast(s)
    weeks = list(range_epiweeks(first_week, last_week, inclusive=True))
    nowcasts = Nowcast(self.data_source).batch_nowcast(weeks)

    # save to database
    with self.database as db:

      # save each nowcast
      for week, nowcast in zip(weeks, nowcasts):
        for location, value, stdev in nowcast:
          db.insert(week, location, float(value), float(stdev))

      # update the timestamp
      db.set_last_update_time()


def get_argument_parser():
  """Define command line arguments and usage."""
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--first',
      type=int,
      help='nowcast a range of weeks, starting with this one')
  parser.add_argument(
      '--last',
      type=int,
      help='nowcast a range of weeks, ending with this one')
  parser.add_argument(
      '--test',
      action='store_true',
      help='generate a nowcast but do not write it to the database')
  return parser


def validate_args(args):
  """Validate and return command line arguments."""
  if (args.first is None) ^ (args.last is None):
    raise Exception('`first` and `last` must be used together')
  if args.first and args.first > args.last:
    raise Exception('`first` must be less than or equal to `last`')
  return args.first, args.last, args.test


def main(first, last, test):
  """Run this script from the command line."""
  NowcastUpdate.new_instance(test).update(first, last)


if __name__ == '__main__':
  main(*validate_args(get_argument_parser().parse_args()))
