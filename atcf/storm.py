"""
Tools for parsing ATCF files
"""

from datetime import datetime, timedelta
import logging
import numpy as np

__all__ = ['Storm', 'ModelForecast', 'word_numbers', 'titles', 'basinletters']

logger = logging.getLogger(__name__)

word_numbers = {
    1: 'ONE', 2: 'TWO', 3: 'THREE', 4: 'FOUR', 5: 'FIVE', 6: 'SIX', 7: 'SEVEN',
    8: 'EIGHT', 9: 'NINE', 10: 'TEN', 11: 'ELEVEN', 12: 'TWELVE', 13: 'THIRTEEN',
    14: 'FOURTEEN', 15: 'FIFTEEN', 16: 'SIXTEEN', 17: 'SEVENTEEN', 18: 'EIGHTEEN',
    19: 'NINETEEN', 20: 'TWENTY', 21: 'TWENTYONE', 22: 'TWENTYTWO', 23: 'TWENTYTHREE',
    24: 'TWENTYFOUR', 25: 'TWENTYFIVE', 26: 'TWENTYSIX', 27: 'TWENTYSEVEN',
    28: 'TWENTYEIGHT', 29: 'TWENTYNINE', 30: 'THIRTY',
}

# Dictionary of titles for disturbances/storms based on ATCF abbreviations
titles = {
    'WV': 'Invest', 'LO': 'Invest', 'DB': 'Invest',
    'TD': 'Tropical Depression', 'TS': 'Tropical Storm',
    'HU': 'Hurricane', 'SD': 'Subtropical Depression',
    'SS': 'Subtropical Storm', 'PT': 'Post-tropical Storm',
    'EX': 'Extratropical Storm', 'XX': 'Unknown Storm'
}

# Map basin IDs to the basin letters used in storm IDs
basinletters = {'AL': 'L', 'LS': 'Q', 'EP': 'E', 'CP': 'C'}


class Storm:
    """
    TC object constructed from ATCF data

    Args:
        data: Contents of a best track ATCF file
    """
    def __init__(self, data):
        lines = [l.strip() for l in data.split('\n') if len(l.strip()) > 0]
        # Group lines by time. If multiple wind radii thresholds exist,
        # multiple lines will exist for each ob
        timegroups = {}
        for line in lines:
            timestr = line.split(',')[2].strip()
            timegroups.setdefault(timestr, []).append(line)
        # Parse observations from each line of the file, which is ordered by time
        self.obs = []
        for group in timegroups.values():
            try:
                ob = BestTrackObservation(group)
            except:
                logger.exception('error parsing best track line; skipping')
            else:
                self.obs.append(ob)
        self.obs_by_time = {ob.time: ob for ob in self.obs}
        # Define storm attributes from the latest observation
        for var, value in vars(self.obs[-1]).items():
            setattr(self, var, value)
        # Define storm history based on all best track observations
        self.history = StormHistory(self)
        # Time-invariant attributes
        fields = [s.strip() for s in lines[-1].split(',')]
        self.number = int(fields[1]) # Storm ID number, 90+ for invests, 01+ for TCs
        # Full ID includes the basin identifier for the originating basin
        self.ID = '{0:02.0f}'.format(self.number) + basinletters.get(self.history.basin[0], '')

        # Storm title
        if self.number >= 90:
            self.title = 'Invest'
        elif self.classification in titles and self.number < 70:
            # If system has not been a TC before, this is a potential TC
            if not any(c in self.history.classification for c in ('TD','TS','HU','SD','SS','PT')):
                self.title = 'Potential Tropical Cyclone'
            # The system has been (and maybe still is) a storm
            else:
                # If system is no longer a storm, don't call it "invest"
                if self.classification not in ('TD','TS','HU','SD','SS','PT'):
                    self.title = 'Remnants of'
                else:
                    self.title = titles[self.classification]
        else:
            self.title = 'Unknown Storm'

        # Storm name
        name = lines[-1].split(',')[27].strip()
        if self.title == 'Potential Tropical Cyclone':
            # The name is the number in word form. Sometimes this is mislabeled
            # in the file, so contruct it manually.
            self.name = word_numbers.get(self.number, name)
        else:
            # If NHC errantly labels a new TC as "INVEST", use the number name
            # as a better alternative
            if self.number < 90 and name == 'INVEST':
                self.name = word_numbers.get(self.number, name)
            else:
                self.name = name

        # Determine if an invest has become a (S)TC
        transinfo = [line.split('TRANSITIONED,')[-1].split(',')[0].strip()
                     for line in lines if 'TRANSITIONED' in line]
        # Get storm ID number from the latest transition entry
        if transinfo:
            self.transID = transinfo[-1].split()[-1][2:4] + basinletters.get(self.basin, '')
            self.oldID = '9' + transinfo[-1].split()[-3][3] + basinletters.get(self.basin, '')
        else:
            self.transID = None
            self.oldID = None

        # A place to store the latest model data for the storm
        # so that it can be accessed repeatedly after being downloaded once
        self.modeldata = None

    def __eq__(self, other):
        # Consider to be the same storm if IDs are equal and current times
        # are not different by more than 30 days, since an ID should occur at
        # most once every several weeks (for invests) or once a year (TCs).
        return (self.ID == other.ID and self.name == other.name and
                abs((self.time-other.time).total_seconds()) < 30*86400)

    def __hash__(self):
        return hash((self.ID, self.name, self.time))


class BestTrackObservation:
    """
    Define a single best track observation from ATCF B-Deck files.

    Args:
        lines: list(str) List of lines from an ATCF B-deck file describing one observation
    """
    def __init__(self, lines):
        # For most obs, can use any of the lines from the time group
        fields = [s.strip() for s in lines[0].split(',')]
        self.basin = fields[0] # Two-letter ocean basin designation
        self.time = datetime.strptime(fields[2], '%Y%m%d%H') # Observation time
        # Coordinates (longitude in [0, 360])
        lat, lon = fields[6], fields[7]
        if lat[-1] == 'N':
            self.lat = 0.1*float(lat[:-1])
        elif lat[-1] == 'S':
            self.lat = -0.1*float(lat[:-1])
        if lon[-1] == 'W':
            self.lon = 360 - 0.1*float(lon[:-1])
        elif lon[-1] == 'E':
            self.lon = 0.1*float(lon[:-1])
        self.maxwind = int(fields[8]) # Maximum sustained wind speed (kt)
        self.pres = int(fields[9]) if int(fields[9]) > 0 else np.nan # Central pressure (hPa)
        self.classification = fields[10] # 2-letter storm classification
        # Wind radii (nm) of 34 kt, 50 kt, or 64 kt, dependent on the threshold(s) specified
        self.wind_radii = {}
        for line in lines:
            lfields = [s.strip() for s in line.split(',')]
            threshold = int(lfields[11]) # kt
            self.wind_radii[threshold] = {'NE': int(lfields[13]), 'SE': int(lfields[14]),
                                          'SW': int(lfields[15]), 'NW': int(lfields[16])}
        self.envpres = int(fields[17]) if fields[17] else np.nan # Last closed isobar (hPa)
        self.roci = int(fields[18]) if fields[18] else np.nan # Radius of outermost closed isobar (nm)
        self.rmw = int(fields[19]) if fields[19] else np.nan # Radius of maximum wind (nm)
        self.maxgust = int(fields[20]) if fields[20] not in ('','0') else np.nan # Maximum wind gust (kt)
        self.eyeD = int(fields[21]) if fields[21] not in ('','0') else np.nan # Eye diameter (nm)


class StormHistory:
    """Namespace for aggregated arrays of storm attributes from the best track history"""
    def __init__(self, storm):
        # Collect attributes for all best track obs
        for var in vars(storm.obs[-1]):
            allvals = [getattr(ob, var) for ob in storm.obs]
            # Use arrays for numerical data
            if type(allvals[0]) in (int, float):
                setattr(self, var, np.array(allvals))
            else:
                setattr(self, var, allvals)


class ModelForecast:
    """
    Class describing model forecasts of track and intensity for TCs

    Args:
        data: Contents of an ATCF A-deck file
    """
    def __init__(self, data):
        lines = [l.strip() for l in data.split('\n') if len(l.strip()) > 0]
        fields = [s.strip() for s in lines[0].split(',')]
        self.modelname = fields[4]
        # Initialization time
        self.init = datetime.strptime(fields[2], '%Y%m%d%H')
        # Parse time-dependent attributes
        dtypes = {
            'lat': float, 'lon': float, 'FH': float, 'time': 'datetime64[m]',
            'wind': float, 'mslp': float, 'rmw': float,
        }
        # Initialize arrays
        for arrname, dtype in dtypes.items():
            if dtype is float:
                setattr(self, arrname, np.full(len(lines), fill_value=np.nan, dtype=dtype))
            else:
                setattr(self, arrname, np.empty(len(lines), dtype=dtype))
        for i, line in enumerate(lines):
            fields = [field.strip() for field in line.split(',')]
            # Storm center coordinates (lon taken in degrees east (-180 to 180))
            latstr, lonstr = fields[6], fields[7]
            if latstr[-1] == 'N':
                lat = 0.1*float(latstr[:-1])
            elif lat[-1] == 'S':
                lat = -0.1*float(latstr[:-1])
            if lonstr[-1] == 'W':
                lon = -0.1*float(lonstr[:-1])
            elif lonstr[-1] == 'E':
                lon = 0.1*float(lonstr[:-1])
            self.lat[i] = lat
            self.lon[i] = lon
            # Forecast hour
            self.FH[i] = int(fields[5])
            # Forecast valid time
            ftime = self.init + timedelta(hours=self.FH[i])
            self.time[i] = ftime
            # Maximum wind (kt)
            wind = float(fields[8] or np.nan)
            self.wind[i] = wind if wind > 0 else np.nan
            # MSLP in hPa
            mslp = float(fields[9] or np.nan)
            self.mslp[i] = mslp if mslp > 0 else np.nan
            rmw = float(fields[19] or np.nan)
            self.rmw[i] = rmw if rmw > 0 else np.nan
        # Post-process longitudes to make sure plots work correctly.
        # If track crosses prime meridian, turn 0 longitude into 360
        # The 1st condition below can be met at either prime meridian or dateline.
        # The 2nd condition ensures we only catch the prime meridian
        # (storm crossing dateline will have max lon near 180, and it can't jump
        # 40 degrees between two track points to cross that line)
        minlon, maxlon = min(self.lon), max(self.lon)
        if minlon*maxlon <= 0 and abs(maxlon) < 140:
           self.lon = [lon+360 if lon >= 0 else lon for lon in self.lon]
        # Make sure lon is defined in [0,360] not [-180,180] to avoid problems across dateline
        self.lon = [lon+360 if lon < 0 else lon for lon in self.lon]
