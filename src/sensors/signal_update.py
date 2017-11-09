"""
===============
=== Purpose ===
===============

Computes an ILI signal from each of our various digital surveillance sources.
This is done by training a linear regression model with final (stable) FluView
wILI. For any given epiweek, the training set is the preceding K(+3) weeks,
where K is defined apriori as 52 weeks. In other words, the training set is a
sliding window of 52 weeks. The window is lagged by 3 weeks because anything
more recent is assumed to be too unstable due to backfill.

Google no longer produces GFT, so the signal is no longer available in
real-time.

Each signal is updated over the following inclusive range of epiweeks:
  - epiweek of most recently computed signal of this type
  - last epiweek
The idea is to recompute the last stored value (just in case there were
changes to the underlying data source), and to compute all weeks up to, but
not including, the current week (because the current week is, by definition,
still ongoing).

The following signals are (or will eventually be) available:
  - 'gft': based on Google Flu Trends
  - 'ght': based on Google Health Trends
  - 'twitter': based on Health Tweets
  - 'wiki': based on wikipedia access
  - 'uili': autoregression on unstable ILI

See also:
  - http://research.undefinedx.com/forum/index.php?topic=309.0
  - http://research.undefinedx.com/forum/index.php?topic=310.0
  - gft_update.py
  - ght_update.py
  - twitter_update.py
  - wiki.py


=======================
=== Data Dictionary ===
=======================

`signals` is the table where the data is stored.
+----------+-------------+------+-----+---------+----------------+
| Field    | Type        | Null | Key | Default | Extra          |
+----------+-------------+------+-----+---------+----------------+
| id       | int(11)     | NO   | PRI | NULL    | auto_increment |
| name     | varchar(16) | NO   | MUL | NULL    |                |
| epiweek  | int(11)     | NO   | MUL | NULL    |                |
| location | varchar(8)  | NO   | MUL | NULL    |                |
| value    | float       | NO   |     | NULL    |                |
+----------+-------------+------+-----+---------+----------------+
id: unique identifier for each record
name: the name of the signal (ex: 'wiki')
epiweek: the epiweek during which the data was collected
location: where the data was collected (see below)
value: the estimated final value of wILI on this epiweek

There are 62 locations for 'gft' and 'twitter':
  - 'nat' (U.S. National): 1
  - 'hhs[1-10]' (HHS regions): 10
  - '[two-letter state]' (U.S. states and DC): 51

There are 11 locations for 'uili':
  - 'nat' (U.S. National): 1
  - 'hhs[1-10]' (HHS regions): 10

There is 1 location for 'wiki' and 'ght':
  - 'nat' (U.S. National): 1


=================
=== Changelog ===
=================

2016-12-13
  + use secrets
2016-07-17
  + added tracebacks for easier debugging
2016-01-29
  + added `ght` signal
2015-12-11
  + added `wiki` and `uili` signals
  * rewrote `gft` and `twitter` signal
2015-09-07
  * original version (`gft` finished; `twitter` in progress)
2015-09-04
  * work in progress
"""

# built-in libraries
import traceback
# external libraries
import mysql.connector
from sklearn.linear_model import LinearRegression
# local files
from delphi_epidata import Epidata
import fluv_utils as flu
import secrets

# global constants
STATES = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL', 'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME', 'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH', 'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']
REGIONS = ['hhs1', 'hhs2', 'hhs3', 'hhs4', 'hhs5', 'hhs6', 'hhs7', 'hhs8', 'hhs9', 'hhs10']
NATIONAL = ['nat']
TRAIN_LAG = 3
TRAIN_SIZE = 52

# wiki constants
ARTICLES = ['human_flu', 'influenza', 'influenza_a_virus', 'influenzavirus_a', 'influenzavirus_c', 'oseltamivir', 'zanamivir']
HOURS = [17, 18, 21]

# uili constants
UILI_WEEKS = 3


def sql_epiweek(cur, sql, default=None):
  cur.execute(sql)
  epiweek = None
  for (epiweek,) in cur:
    pass
  return default if epiweek is None else epiweek


def get_model(X, Y, fit_intercept=True):
  lr = LinearRegression(fit_intercept=fit_intercept)
  lr.fit(X, Y)
  if fit_intercept:
    return [lr.intercept_[0]] + [b for b in lr.coef_[0]]
  else:
    return [0] + [b for b in lr.coef_[0]]


def predict(model, features):
  return float(model[0] + sum([a * b for a, b in zip(model[1:], features)]))


def api_fetch(res):
  if res['result'] != 1:
    raise Exception('API result=%d (%s)' % (res['result'], res['message']))
  return res['epidata']


def get_ili(location, issue, ew1, ew2):
  result = {}
  epiweeks = Epidata.range(ew1, ew2)
  num_weeks = flu.delta_epiweeks(ew1, ew2) + 1
  # try to get unstable, but gracefully fall back to stable
  if issue is not None:
    res = Epidata.fluview(location, epiweeks, issues=issue)
    if res['result'] == 1:
      for row in res['epidata']:
        result[row['epiweek']] = row['wili']
  # check to see if another API call is needed
  if issue is None or res['result'] != 1 or len(res['epidata']) < num_weeks:
    # get stable data
    data = api_fetch(Epidata.fluview(location, epiweeks))
    for row in data:
      epiweek = row['epiweek']
      if epiweek not in result:
        result[epiweek] = row['wili']
  # return a list of weekly data
  return [[result[ew]] for ew in sorted(list(result.keys()))]


def get_gft(location, ew1, ew2):
  epiweeks = Epidata.range(ew1, ew2)
  data = api_fetch(Epidata.gft(location, epiweeks))
  return [[1e-3 * row['num']] for row in data]


def get_ght(ew1, ew2):
  epiweeks = Epidata.range(ew1, ew2)
  data = api_fetch(Epidata.ght(secrets.api.ght, 'US', epiweeks, '/m/0cycc'))
  return [[row['value']] for row in data]


def get_twitter(location, ew1, ew2):
  epiweeks = Epidata.range(ew1, ew2)
  data = api_fetch(Epidata.twitter(secrets.api.twitter, location, epiweeks=epiweeks))
  return [[row['percent']] for row in data]


def get_wiki(ew1, ew2):
  # get the raw wiki data, broken down by epiweek, article, and hour
  epiweeks = Epidata.range(ew1, ew2)
  result = {}
  data = api_fetch(Epidata.wiki(ARTICLES, epiweeks=epiweeks, hours=HOURS))
  # index the data for fast access
  for row in data:
    epiweek, article = row['epiweek'], row['article']
    if epiweek not in result:
      result[epiweek] = {}
    if article not in result[epiweek]:
      result[epiweek][article] = {'c': [], 't': []}
    result[epiweek][article]['c'].append(row['count'])
    result[epiweek][article]['t'].append(row['total'])
  # group by epiweek and article (combining hours)
  data = []
  for epiweek in sorted(list(result.keys())):
    row = []
    for article in sorted(ARTICLES):
      count, total = result[epiweek][article]['c'], result[epiweek][article]['t']
      if len(count) != len(HOURS) or len(total) != len(HOURS):
        raise Exception('wiki is missing hours')
      row.append(1e6 * sum(count) / sum(total))
    data.append(row)
  # return a list of weekly data
  return data


def get_uili(location, ew1, ew2):
  # predict stable ili using UILI_WEEKS(=3) most recent unstable ili
  uili = []
  for ew in flu.range_epiweeks(ew1, ew2, inclusive=True):
    ew_train_start = flu.add_epiweeks(ew, -UILI_WEEKS)
    ew_train_end = flu.add_epiweeks(ew, -1)
    data = get_ili(location, ew_train_end, ew_train_start, ew_train_end)
    if len(data) != UILI_WEEKS:
      raise Exception('invalid uili@%d data [%d-%d=>%d]' % (ew_train_end, ew_train_start, ew_train_end, ew))
    uili.append([row[0] for row in data])
  return uili


def get_update_weeks(epiweek):
  # `epiweek` (aka `week4`): want to produce the signal for this week
  # `week1` - `week2`: training range (inclusive)
  # `week3`: fetch fluview data as of this issue (the week before `epiweek`)
  week1 = flu.add_epiweeks(epiweek, -(TRAIN_LAG + TRAIN_SIZE))
  week2 = flu.add_epiweeks(epiweek, -(TRAIN_LAG + 1))
  week3 = flu.add_epiweeks(epiweek, -1)
  return week1, week2, week3


def extract_signal(train_x, train_y, test, week4):
  # data timing
  week1, week2, week3 = get_update_weeks(week4)
  # check the data
  if len(train_x) != TRAIN_SIZE:
    raise Exception('invalid training source data [%d-%d]' % (week1, week2))
  if len(train_y) != TRAIN_SIZE:
    raise Exception('invalid training fluview@%d data [%d-%d]' % (week3, week1, week2))
  if len(test) != 1:
    raise Exception('invalid testing source data [%d]' % week4)
  # fit the regression model
  model = get_model(train_x, train_y)
  # get the signal
  return predict(model, test[0])


def update_gft(cur, last_epiweek):
  # hardcoded first week in case the database is empty (first run)
  first_epiweek = flu.add_epiweeks(200340, TRAIN_SIZE + TRAIN_LAG)
  # lookup the week of the most recently calculated signal
  sql = "SELECT max(`epiweek`) FROM `signals` WHERE `name` = 'gft'"
  first_epiweek = min(sql_epiweek(cur, sql, first_epiweek), last_epiweek)
  # find out much work there is to do
  num_weeks = flu.delta_epiweeks(first_epiweek, last_epiweek) + 1
  print(' gft available through %d: updating %d week(s)' % (first_epiweek, num_weeks))
  # update each week
  sql = "INSERT INTO `signals` (`name`, `epiweek`, `location`, `value`) VALUES ('gft', %s, %s, %s) ON DUPLICATE KEY UPDATE `value` = %s"
  for week4 in flu.range_epiweeks(first_epiweek, last_epiweek, inclusive=True):
    week1, week2, week3 = get_update_weeks(week4)
    print('  as of %d: training on %d-%d to predict %d' % (week3, week1, week2, week4))
    # update each location
    #for location in NATIONAL + REGIONS + STATES:
    for location in NATIONAL + REGIONS:
      # gather all the data
      gft_train = get_gft(location, week1, week2)
      ili_train = get_ili(location, week3, week1, week2)
      gft_test = get_gft(location, week4, week4)
      # get the signal
      signal = extract_signal(gft_train, ili_train, gft_test, week4)
      print('   %s: %.3f' % (location, signal))
      # save to database
      cur.execute(sql, (week4, location, signal, signal))


def update_ght(cur, last_epiweek):
  # hardcoded first week in case the database is empty (first run)
  first_epiweek = flu.add_epiweeks(200401, TRAIN_SIZE + TRAIN_LAG)
  # lookup the week of the most recently calculated signal
  sql = "SELECT max(`epiweek`) FROM `signals` WHERE `name` = 'ght'"
  first_epiweek = min(sql_epiweek(cur, sql, first_epiweek), last_epiweek)
  # find out much work there is to do
  num_weeks = flu.delta_epiweeks(first_epiweek, last_epiweek) + 1
  print(' ght available through %d: updating %d week(s)' % (first_epiweek, num_weeks))
  # update each week
  sql = "INSERT INTO `signals` (`name`, `epiweek`, `location`, `value`) VALUES ('ght', %s, %s, %s) ON DUPLICATE KEY UPDATE `value` = %s"
  for week4 in flu.range_epiweeks(first_epiweek, last_epiweek, inclusive=True):
    week1, week2, week3 = get_update_weeks(week4)
    print('  as of %d: training on %d-%d to predict %d' % (week3, week1, week2, week4))
    # ght is only available nationally until Google fixes the API
    location = NATIONAL[0]
    # gather all the data
    ght_train = get_ght(week1, week2)
    ili_train = get_ili(location, week3, week1, week2)
    ght_test = get_ght(week4, week4)
    # get the signal
    signal = extract_signal(ght_train, ili_train, ght_test, week4)
    print('   %s: %.3f' % (location, signal))
    # save to database
    cur.execute(sql, (week4, location, signal, signal))


def update_twitter(cur, last_epiweek):
  # hardcoded first week in case the database is empty (first run)
  first_epiweek = flu.add_epiweeks(201148, TRAIN_SIZE + TRAIN_LAG)
  # lookup the week of the most recently calculated signal
  sql = "SELECT max(`epiweek`) FROM `signals` WHERE `name` = 'twitter'"
  first_epiweek = min(sql_epiweek(cur, sql, first_epiweek), last_epiweek)
  # find out much work there is to do
  num_weeks = flu.delta_epiweeks(first_epiweek, last_epiweek) + 1
  print(' twitter available through %d: updating %d week(s)' % (first_epiweek, num_weeks))
  # update each week
  sql = "INSERT INTO `signals` (`name`, `epiweek`, `location`, `value`) VALUES ('twitter', %s, %s, %s) ON DUPLICATE KEY UPDATE `value` = %s"
  for week4 in flu.range_epiweeks(first_epiweek, last_epiweek, inclusive=True):
    week1, week2, week3 = get_update_weeks(week4)
    print('  as of %d: training on %d-%d to predict %d' % (week3, week1, week2, week4))
    # update each location
    #for location in NATIONAL + REGIONS + STATES:
    for location in NATIONAL + REGIONS:
      # gather all the data
      twitter_train = get_twitter(location, week1, week2)
      ili_train = get_ili(location, week3, week1, week2)
      twitter_test = get_twitter(location, week4, week4)
      # get the signal
      signal = extract_signal(twitter_train, ili_train, twitter_test, week4)
      print('   %s: %.3f' % (location, signal))
      # save to database
      cur.execute(sql, (week4, location, signal, signal))


def update_wiki(cur, last_epiweek):
  # hardcoded first week in case the database is empty (first run)
  first_epiweek = flu.add_epiweeks(201030, TRAIN_SIZE + TRAIN_LAG)
  # lookup the week of the most recently calculated signal
  sql = "SELECT max(`epiweek`) FROM `signals` WHERE `name` = 'wiki'"
  first_epiweek = min(sql_epiweek(cur, sql, first_epiweek), last_epiweek)
  # find out much work there is to do
  num_weeks = flu.delta_epiweeks(first_epiweek, last_epiweek) + 1
  print(' wiki available through %d: updating %d week(s)' % (first_epiweek, num_weeks))
  # update each week
  sql = "INSERT INTO `signals` (`name`, `epiweek`, `location`, `value`) VALUES ('wiki', %s, %s, %s) ON DUPLICATE KEY UPDATE `value` = %s"
  for week4 in flu.range_epiweeks(first_epiweek, last_epiweek, inclusive=True):
    week1, week2, week3 = get_update_weeks(week4)
    print('  as of %d: training on %d-%d to predict %d' % (week3, week1, week2, week4))
    # wiki doesn't really have a location, just assume U.S. National
    location = NATIONAL[0]
    # gather all the data
    wiki_train = get_wiki(week1, week2)
    ili_train = get_ili(location, week3, week1, week2)
    wiki_test = get_wiki(week4, week4)
    # get the signal
    signal = extract_signal(wiki_train, ili_train, wiki_test, week4)
    print('   %s: %.3f' % (location, signal))
    # save to database
    cur.execute(sql, (week4, location, signal, signal))


def update_uili(cur, last_epiweek):
  # hardcoded first week in case the database is empty (first run)
  first_epiweek = flu.add_epiweeks(200950, TRAIN_SIZE + TRAIN_LAG)
  # lookup the week of the most recently calculated signal
  sql = "SELECT max(`epiweek`) FROM `signals` WHERE `name` = 'uili'"
  first_epiweek = min(sql_epiweek(cur, sql, first_epiweek), last_epiweek)
  # find out much work there is to do
  num_weeks = flu.delta_epiweeks(first_epiweek, last_epiweek) + 1
  print(' uili available through %d: updating %d week(s)' % (first_epiweek, num_weeks))
  # update each week
  sql = "INSERT INTO `signals` (`name`, `epiweek`, `location`, `value`) VALUES ('uili', %s, %s, %s) ON DUPLICATE KEY UPDATE `value` = %s"
  for week4 in flu.range_epiweeks(first_epiweek, last_epiweek, inclusive=True):
    week1, week2, week3 = get_update_weeks(week4)
    print('  as of %d: training on %d-%d to predict %d' % (week3, week1, week2, week4))
    # update each location
    for location in NATIONAL + REGIONS:
      # gather all the data
      uili_train = get_uili(location, week1, week2)
      ili_train = get_ili(location, week3, week1, week2)
      uili_test = get_uili(location, week4, week4)
      # get the signal
      signal = extract_signal(uili_train, ili_train, uili_test, week4)
      print('   %s: %.3f' % (location, signal))
      # save to database
      cur.execute(sql, (week4, location, signal, signal))


def main():
  # database setup
  u, p = secrets.db.epi
  cnx = mysql.connector.connect(user=u, password=p, database='epidata')
  cur = cnx.cursor()

  # find the most recent completed epiweek (last week)
  last_epiweek = sql_epiweek(cur, 'SELECT yearweek(date_sub(now(), INTERVAL 1 WEEK), 6)')
  print('Updating signals through %d' % last_epiweek)

  # update each source
  exceptions = []
  #try:
  #  update_gft(cur, last_epiweek)
  #except Exception as e:
  #  exceptions.append(e)
  try:
    update_ght(cur, last_epiweek)
  except Exception as e:
    print('ght exception')
    print(traceback.format_exc())
    exceptions.append(e)
  try:
    update_twitter(cur, last_epiweek)
  except Exception as e:
    print('twitter exception')
    print(traceback.format_exc())
    exceptions.append(e)
  try:
    update_wiki(cur, last_epiweek)
  except Exception as e:
    print('wiki exception')
    print(traceback.format_exc())
    exceptions.append(e)
  try:
    update_uili(cur, last_epiweek)
  except Exception as e:
    print('uili exception')
    print(traceback.format_exc())
    exceptions.append(e)

  # database cleanup
  cur.close()
  cnx.commit()
  cnx.close()

  # check for problems
  if len(exceptions) != 0:
    for ex in exceptions:
      print(ex)
    raise Exception(exceptions)


if __name__ == '__main__':
  main()
