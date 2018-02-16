"""
===============
=== Purpose ===
===============

Prepares the variables necessary for performing sensor fusion of signals from
the various regions and states within the US. This includes, in particular, the
matrices H and W.

H maps from state space (columns) to input space (rows). W maps from state
space (columns) to output space (rows).

In general, this file takes as input the locations for which sensor readings
are available and returns as output H, W, and a list of locations which make up
output space.
"""

# standard library
from fractions import Fraction
import functools

# third party
import numpy as np

# first party
import delphi.nowcast.fusion.fusion as fusion
from delphi.utils.geo.locations import Locations
from delphi.utils.geo.populations import get_population


class UsFusion:
  """Prepares for sensor fusion of signals based on US regions and states."""

  __known_statespace = {}

  @staticmethod
  def get_weight_row(location, season, atoms):
    """
    Return a list of the population weights of all atoms, with respect to the
    given location. Atoms not within the location will have a weight of zero.
    The returned weights will sum to one.
    """

    # check each atom individually
    total_population = 0
    atom_populations = []
    for atom in atoms:
      if atom in Locations.region_map[location]:
        population = get_population(atom, season)
      else:
        population = 0
      total_population += population
      atom_populations.append(population)

    # sanity check
    if total_population == 0:
      raise Exception(('location has no constituent atoms', location))

    # return list of fractional populations
    get_fraction = lambda pop: Fraction(pop, total_population)
    return list(map(get_fraction, atom_populations))

  @staticmethod
  def get_weight_matrix(locations, season, atoms):
    """
    Return a matrix of weights, where rows correspond to the given locations
    and columns correspond to the given atomic locations.
    """

    # stack rows for each location and return the matrix
    get_row = lambda loc: UsFusion.get_weight_row(loc, season, atoms)
    return np.array(list(map(get_row, locations)))

  @staticmethod
  @functools.lru_cache(maxsize=16)
  def determine_statespace(
    input_locations,
    season=None,
    exclude_locations=()
  ):
    """
    Return matrices mapping from latent statespace to input space and output
    space. These are the matrices H and W, respectively, used in the sensor
    fusion kernel. A list of output locations corresponding to the rows of W is
    also returned.

    Results are cached for better performance.

    inputs:
      input_locations: a tuple of sensor locations
      season (optional): The season (year) in which the nowcast is being made.
        This is generally only helpful for retrospective nowcasts where
        historical population weights should be used. By default, the most
        recent population weights are used. (See populations.py)
      exclude_locations (optional): A tuple of atoms to exclude from
        statespace. This is generally only helpful for retrospective nowcasts
        where it is known that some state or territory was not reporting and,
        therefore, was not included in regional or national wILI.

    outputs:
      - the matrix H, mapping subspace to inputs
      - the matrix W, mapping subspace to outputs
      - tuple of output locations, corresponding to rows of W
    """

    # quick sanity check
    if set(exclude_locations) & set(input_locations):
      raise Exception('input contains excluded locations')

    # function to filter out excluded atoms
    atom_filter = lambda a: a not in exclude_locations

    # list of all locations, including nat, hhs, cen, and atoms
    all_locations = list(filter(atom_filter, Locations.region_list))

    # list of atomic locatinos only
    atoms = list(filter(atom_filter, Locations.atom_list))

    # precursors of the H and W matrices, assuming that statespace is US atoms
    get_matrix = lambda locs: UsFusion.get_weight_matrix(locs, season, atoms)
    H0 = get_matrix(input_locations)
    W0 = get_matrix(all_locations)

    # optimization for the typical case where all US atoms are represented
    if set(input_locations) >= set(atoms):
      # statespace is all US atoms, so H and W are already correct
      H, W, output_locations = H0, W0, all_locations
    else:
      # determine optimal H and W matrices
      H, W, selected_rows = fusion.determine_statespace(H0, W0)
      # select the output locations
      output_locations = [all_locations[i] for i in selected_rows]

    # convert fractions to floats and return the result
    return H.astype(np.float), W.astype(np.float), output_locations
