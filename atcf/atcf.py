"""
pyGPLOT/fileio/atcf.py

Author: Levi Cowan

DESCRIPTION
-----------
    Tools for parsing ATCF files

CLASSES
-------
    ADeck:         Collect and parse entries from an ATCF A-deck file

    ADeckEntry:    Define a single entry from an ATCF A-deck file

    ModelForecast: Model forecast object constructed from ATCF A-deck data,
                   containing time-ordered arrays of storm attributes

    BDeck:         Collect and parse entries from an ATCF B-deck file

    BDeckEntry:    Define a single entry from an ATCF B-deck file.

    Storm:         Storm object constructed from ATCF B-deck data,
                   containing time-ordered arrays of storm attributes
"""
__all__ = ['ADeck', 'BDeck', 'ModelForecast', 'Storm']

from datetime import datetime, timedelta
import logging
import numpy as np

logger = logging.getLogger(__name__)


# Map basin IDs to the basin letters used in storm IDs
basinletters = {'AL': 'L', 'LS': 'Q', 'EP': 'E', 'CP': 'C', 'WP': 'W'}


class ADeck:
    """
    Collect and parse entries from an ATCF A-deck file

    Attributes
    ----------
        lines:           [list(str)]        List of lines from the input file

        line_timegroups: [dict(str,list)]   Groups of lines from the input file
                                            associated with a single time/entry

        entries:         [list(ADeckEntry)] List of ADeckEntry objects

        entries_by_modelrun: [dict(tuple, list(ADeckEntry))]
            Maps each (stormID, modelname, init) tuple to a list of ADeckEntry objects

    Methods
    -------
        get_forecast(stormID=None, modelname=None, init=None):
            Construct a ModelForecast object for a single storm and model run.
            If no specific storm, model, or init is requested and the A-deck
            includes only one choice, the forecast for that choice is constructed.

        get_all_forecasts():
            Construct a list of ModelForecast objects for all forecasts in this A-deck
    """
    # Map attribute names to their field index in a line from an A-deck file
    attr_indices = {
        'basin': 0, 'number': 1, 'init': 2, 'modelname': 4, 'fhour': 5, 'lat': 6,
        'lon': 7, 'vmax': 8, 'pmin': 9, 'R34_NE': 13, 'R34_SE': 14, 'R34_SW': 15,
        'R34_NW': 16, 'R50_NE': 13, 'R50_SE': 14, 'R50_SW': 15, 'R50_NW': 16,
        'R64_NE': 13, 'R64_SE': 14, 'R64_SW': 15, 'R64_NW': 16, 'poci': 17,
        'roci': 18, 'rmw': 19
    }

    # Possible missing values for whitespace-stripped fields
    missing_values = ['', '-99', '-999']

    def __init__(self, filename):
        """
        Args
        ----
            filename: [str] ATCF A-deck filename
        """
        self.filename = filename
        with open(self.filename) as f:
            self.lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
        # Tag each line with its storm ID, model name, init time, and forecast hour
        lineIDs = {}
        for line in self.lines:
            fields = [f.strip() for f in line.split(',')]
            stormnum = fields[self.attr_indices['number']]
            modelname = fields[self.attr_indices['modelname']]
            init = datetime.strptime(fields[self.attr_indices['init']], '%Y%m%d%H')
            fhour = int(fields[self.attr_indices['fhour']])
            lineIDs[(stormnum, modelname, init, fhour)] = line
        # Group lines by time for each storm and model run. If multiple wind radii
        # thresholds exist, multiple lines will exist for each lead time
        self.line_timegroups = {
            (stormnum, modelname, init): {} for stormnum, modelname, init
            in set((ID, mname, it) for ID, mname, it, _ in lineIDs.keys())
        }
        for ID, line in lineIDs.items():
            stormnum, modelname, init, fhour = ID
            self.line_timegroups[(stormnum, modelname, init)].setdefault(fhour, []).append(line)
        # Parse entries from each line of the file, which is ordered by time
        self.entries = []
        for ID, timegroups in self.line_timegroups.items():
            for group in timegroups.values():
                try:
                    entry = ADeckEntry(group)
                except:
                    logger.exception('error parsing A-deck entry; skipping')
                else:
                    self.entries.append(entry)
        # Group entries by storm and model since multiple storms and/or models
        # may appear in the file
        self.entries_by_modelrun = {}
        for e in self.entries:
            self.entries_by_modelrun.setdefault((e.stormID, e.modelname, e.init), []).append(e)

    def get_forecast(self, stormID=None, modelname=None, init=None):
        """
        Construct a ModelForecast object for a single storm and model run.
        If no specific storm, model, or init is requested and the A-deck
        includes only one choice, the forecast for that choice is constructed.

        Args
        ----
            adeck:     [ADeck]                       ADeck instance
            stormID:   [str]                         Requested storm ID (e.g., '01L')
            modelname: [str]                         Requested model name
            init:      [datetime or str(YYYYMMDDHH)] Requested model initialization time
        """
        return ModelForecast(self, stormID, modelname, init)

    def get_all_forecasts(self):
        """
        Construct a list of ModelForecast objects for all forecasts in this A-deck
        """
        forecasts = []
        for stormID, modelname, init in self.entries_by_modelrun:
            forecasts.append(ModelForecast(self, stormID, modelname, init))
        return forecasts

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        nruns = len(self.entries_by_modelrun.keys())
        nstorms, nmodels, _ = map(len, map(set, zip(*list(self.entries_by_modelrun.keys()))))
        return (f'<ADeck with {nruns} model runs from {nmodels} models '
                f'for {nstorms} storms>')


class ADeckEntry:
    """
    Define a single entry from an ATCF A-Deck file.

    Attributes
    ----------
        modelname:      [str]      Model name
        stormID:        [str]      Storm ID (e.g., '01L')
        basin:          [str]      2-letter basin IDs at each time (e.g., 'AL')
        init:           [datetime] Model initialization time
        fh:             [int]
        validtime:      [datetime] Forecast valid time
        lat:            [float]    Latitude (deg N)
        lon:            [float]    Longitude ([-180, 180])
        vmax:           [float]    Maximum sustained wind (kt)
        pmin:           [float]    Minimum central pressure (hPa)
        rmw:            [float]    Radius of maximum wind (nm)
        roci:           [float]    Radius of outermost closed isobar (nm)
        poci:           [float]    Pressure of outermost closed isobar (hPa)
        R34_{quad}:     [float]    Radius of 34kt wind (nm) in the `quad` quadrant,
                                   where `quad` is one of ['NE','SE','SW','NW']
        R50_{quad}:     [float]    Radius of 50kt wind (nm) in the `quad` quadrant,
                                   where `quad` is one of ['NE','SE','SW','NW']
        R64_{quad}:     [float]    Radius of 64kt wind (nm) in the `quad` quadrant,
                                   where `quad` is one of ['NE','SE','SW','NW']
    """
    def __init__(self, lines):
        """
        Args
        ----
            lines: [list(str)] List of lines from an ATCF A-deck file describing one entry
        """
        # For most attributes, can use any of the lines from the time group
        fields = [s.strip() for s in lines[0].split(',')]
        self.modelname = fields[ADeck.attr_indices['modelname']]
        # Two-letter ocean basin designation
        self.basin = fields[ADeck.attr_indices['basin']]
        # Storm ID number, 90+ for invests, 01+ for TCs
        self.number = int(fields[ADeck.attr_indices['number']])
        # Full ID includes the basin identifier for the originating basin (e.g., 01L)
        self.stormID = f'{self.number:02}{basinletters[self.basin]}'
        # Initialization time
        self.init = datetime.strptime(fields[ADeck.attr_indices['init']], '%Y%m%d%H')
        # Forecast hour / lead time
        self.fhour = int(fields[ADeck.attr_indices['fhour']])
        self.validtime = self.init + timedelta(hours=self.fhour)
        # Coordinates (longitude in [-180, 180])
        lat, lon = fields[ADeck.attr_indices['lat']], fields[ADeck.attr_indices['lon']]
        if lat[-1] == 'N':
            self.lat = 0.1*float(lat[:-1])
        elif lat[-1] == 'S':
            self.lat = -0.1*float(lat[:-1])
        if lon[-1] == 'W':
            self.lon = -0.1*float(lon[:-1])
        elif lon[-1] == 'E':
            self.lon = 0.1*float(lon[:-1])
        # Maximum sustained wind speed (kt)
        self.vmax = int(fields[ADeck.attr_indices['vmax']])
        # Attributes that may be missing and are positive definite
        for attr in ('pmin', 'poci', 'roci', 'rmw'):
            # These values may not exist in the entry at all
            try:
                valstr = fields[ADeck.attr_indices[attr]]
            except IndexError:
                value = np.nan
            else:
                value = int(valstr) if valstr not in ADeck.missing_values+['0'] else np.nan
            setattr(self, attr, value)
        # Wind radii (nm) of 34 kt, 50 kt, or 64 kt, dependent on the threshold(s) specified
        for line in lines:
            lfields = [s.strip() for s in line.split(',')]
            thresh = int(lfields[11]) # kt
            if thresh == 0:
                continue
            attrs = [f'R{thresh}_{quad}' for quad in ('NE','SE','SW','NW')]
            for attr in attrs:
                setattr(self, attr, int(lfields[ADeck.attr_indices[attr]]))
        # Set wind radii that are not present to missing
        radii_attrs = [f'R{thresh}_{q}' for thresh in (34, 50, 64) for q in ('NE','SE','SW','NW')]
        for attr in radii_attrs:
            if not getattr(self, attr, None):
                setattr(self, attr, np.nan)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f'<ADeckEntry storm={self.stormID} model={self.modelname} '
                f'init={self.init:%Y%m%d%H} fhour={self.fhour}>')


class ModelForecast:
    """
    Model forecast object constructed from ATCF A-deck data, containing time-ordered
    arrays of storm attributes

    Attributes
    ----------
        modelname:      [str]             Model name
        stormID:        [str]             Storm ID (e.g., '01L')
        init:           [datetime]        Model initialization time
        fhour:          [array(int)]      Forecast hour / lead time
        validtime:      [array(datetime)] Forecast valid time
        basin:          [array(str)]      2-letter basin IDs at each time (e.g., 'AL')
        lat:            [array(float)]    Latitude (deg N)
        lon:            [array(float)]    Longitude ([-180, 180])
        vmax:           [array(float)]    Maximum sustained wind (kt)
        pmin:           [array(float)]    Minimum central pressure (hPa)
        rmw:            [array(float)]    Radius of maximum wind (nm)
        roci:           [array(float)]    Radius of outermost closed isobar (nm)
        poci:           [array(float)]    Pressure of outermost closed isobar (hPa)
        R34_{quad}:     [array(float)]    Radius of 34kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R50_{quad}:     [array(float)]    Radius of 50kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R64_{quad}:     [array(float)]    Radius of 64kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
    """
    def __init__(self, adeck, stormID=None, modelname=None, init=None):
        """
        Construct a forecast for a single storm and model run. If no specific storm,
        model, or init is requested and the A-deck includes only one choice, the
        forecast for that choice is constructed.

        Args
        ----
            adeck:     [ADeck]                       ADeck instance
            stormID:   [str]                         Requested storm ID (e.g., '01L')
            modelname: [str]                         Requested model name
            init:      [datetime or str(YYYYMMDDHH)] Requested model initialization time
        """
        # Identify needed A-deck entries
        needed_entries = adeck.entries
        if stormID is None and len(set(e.stormID for e in needed_entries)) > 1:
            raise ValueError('No storm ID requested, but multiple storms exist in A-deck')
        else:
            needed_entries = [e for e in needed_entries if stormID is None
                              or e.stormID == stormID]
        if modelname is None and len(set(e.modelname for e in needed_entries)) > 1:
            raise ValueError('No model requested, but multiple models exist in A-deck')
        else:
            needed_entries = [e for e in needed_entries if modelname is None
                              or e.modelname == modelname]
        init = datetime.strptime(init, '%Y%m%d%H') if type(init) is str else init
        if init is None and len(set(e.init for e in needed_entries)) > 1:
            raise ValueError('No init time requested, but multiple model runs exist in A-deck')
        else:
            needed_entries = [e for e in needed_entries if init is None
                              or e.init == init]
        if not needed_entries:
            raise ValueError(f'No A-deck entries found matching stormID={stormID}, '
                             f'modelname={modelname}, init={init}')

        # Collect time-dependent attributes
        dtypes = {
            'basin': 'U2', 'lat': float, 'lon': float, 'fhour': int,
            'validtime': 'datetime64[m]', 'vmax': float, 'pmin': float,
            'rmw': float, 'poci': float, 'roci': float, 'R34_NE': float, 'R34_SE': float,
            'R34_SW': float, 'R34_NW': float, 'R50_NE': float, 'R50_SE': float,
            'R50_SW': float, 'R50_NW': float, 'R64_NE': float, 'R64_SE': float,
            'R64_SW': float, 'R64_NW': float
        }
        # Create arrays
        for attr, dtype in dtypes.items():
            arr = np.array([getattr(entry, attr) for entry in needed_entries], dtype=dtype)
            setattr(self, attr, arr)

        # Collect time-invariant attributes
        self.modelname = needed_entries[0].modelname
        self.init = needed_entries[0].init
        self.stormID = needed_entries[0].stormID

    def __hash__(self):
        return hash((self.modelname, self.stormID, self.init))

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return (f'<ModelForecast stormID={self.stormID}, model={self.modelname}, '
                f'init={self.init:%Y%m%d%H}>')


class BDeck:
    """
    Collect and parse entries from an ATCF B-deck file

    Attributes
    ----------
        lines:           [list(str)]        List of lines from the input file

        line_timegroups: [dict(str,list)]   Groups of lines from the input file
                                            associated with a single time/entry

        entries:         [list(BDeckEntry)] List of BDeckEntry objects

        entries_by_time: [dict(datetime, BDeckEntry)] Maps each observation time
                                                      to a BDeckEntry object

        stormID:         [str]              Storm ID (e.g., '01L')

    Methods
    -------
        as_storm():
            Construct a Storm object from this B-deck
    """
    # Map attribute names to their field index in a line from an A-deck file
    attr_indices = {
        'basin': 0, 'number': 1, 'time': 2, 'lat': 6, 'lon': 7, 'vmax': 8,
        'pmin': 9, 'classification': 10, 'R34_NE': 13, 'R34_SE': 14,
        'R34_SW': 15, 'R34_NW': 16, 'R50_NE': 13, 'R50_SE': 14,
        'R50_SW': 15, 'R50_NW': 16, 'R64_NE': 13, 'R64_SE': 14,
        'R64_SW': 15, 'R64_NW': 16, 'poci': 17, 'roci': 18,
        'rmw': 19, 'maxgust': 20, 'eye_diameter': 21, 'name': 27
    }

    # Possible missing values for whitespace-stripped fields
    missing_values = ['', '-99', '-999']

    def __init__(self, filename):
        """
        Args
        ----
            filename: [str] ATCF B-deck filename
        """
        self.filename = filename
        with open(self.filename) as f:
            self.lines = [l.strip() for l in f.readlines() if len(l.strip()) > 0]
        # Group lines by time. If multiple wind radii thresholds exist,
        # multiple lines will exist for each ob
        self.line_timegroups = {}
        for line in self.lines:
            timestr = line.split(',')[self.attr_indices['time']].strip()
            self.line_timegroups.setdefault(timestr, []).append(line)
        # Parse entries from each line of the file, which is ordered by time
        self.entries = []
        for group in self.line_timegroups.values():
            try:
                entry = BDeckEntry(group)
            except:
                logger.exception('error parsing B-deck entry; skipping')
            else:
                self.entries.append(entry)
        self.entries_by_time = {entry.time: entry for entry in self.entries}
        self.stormID = self.entries[0].stormID

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'<BDeck for storm={self.stormID} with {len(self.entries)} entries>'

    def as_storm(self):
        """Construct a Storm object from this B-deck"""
        return Storm(self)


class BDeckEntry:
    """
    Define a single entry from an ATCF B-deck file.

    Attributes
    ----------
        stormID:        [str]      Storm ID (e.g., '01L')
        stormname:      [str]      Storm name ('NONAME' if no defined name)
        basin:          [str]      2-letter basin IDs at each time (e.g., 'AL')
        classification: [str]      2-letter storm classification (e.g., 'TS')
        lat:            [float]    Latitude (deg N)
        lon:            [float]    Longitude ([-180, 180])
        time:           [datetime] Observation time
        vmax:           [float]    Maximum sustained wind (kt)
        maxgust:        [float]    Maximum wind gusts (kt)
        pmin:           [float]    Minimum central pressure (hPa)
        rmw:            [float]    Radius of maximum wind (nm)
        roci:           [float]    Radius of outermost closed isobar (nm)
        poci:           [float]    Pressure of outermost closed isobar (hPa)
        eye_diameter:   [float]    Eye diameter (nm)
        R34_{quad}:     [float]    Radius of 34kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R50_{quad}:     [float]    Radius of 50kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R64_{quad}:     [float]    Radius of 64kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
    """
    def __init__(self, lines):
        """
        Args
        ----
            lines: [list(str)] List of lines from an ATCF B-deck file describing one entry
        """
        # For most attributes, can use any of the lines from the time group
        fields = [s.strip() for s in lines[0].split(',')]
        # Two-letter ocean basin designation
        self.basin = fields[BDeck.attr_indices['basin']]
        # Storm ID number, 90+ for invests, 01+ for TCs
        self.number = int(fields[BDeck.attr_indices['number']])
        # Full ID includes the basin identifier for the originating basin (e.g., 01L)
        self.stormID = f'{self.number:02}{basinletters[self.basin]}'
        # Storm name (not always present in the file)
        try:
            self.stormname = fields[BDeck.attr_indices['name']]
        except IndexError:
            self.stormname = 'NONAME'
        # Observation time
        self.time = datetime.strptime(fields[BDeck.attr_indices['time']], '%Y%m%d%H')
        # Coordinates (longitude in [-180, 180])
        lat, lon = fields[BDeck.attr_indices['lat']], fields[BDeck.attr_indices['lon']]
        if lat[-1] == 'N':
            self.lat = 0.1*float(lat[:-1])
        elif lat[-1] == 'S':
            self.lat = -0.1*float(lat[:-1])
        if lon[-1] == 'W':
            self.lon = -0.1*float(lon[:-1])
        elif lon[-1] == 'E':
            self.lon = 0.1*float(lon[:-1])
        # Maximum sustained wind speed (kt)
        self.vmax = int(fields[BDeck.attr_indices['vmax']])
        # Attributes that may be missing and are positive definite
        for attr in ('pmin', 'poci', 'roci', 'rmw', 'maxgust', 'eye_diameter'):
            # These values may not exist in the entry at all
            try:
                valstr = fields[BDeck.attr_indices[attr]]
            except IndexError:
                value = np.nan
            else:
                value = int(valstr) if valstr not in BDeck.missing_values+['0'] else np.nan
            setattr(self, attr, value)
        # 2-letter storm classification
        self.classification = fields[BDeck.attr_indices['classification']]
        # Wind radii (nm) of 34 kt, 50 kt, or 64 kt, dependent on the threshold(s) specified
        for line in lines:
            lfields = [s.strip() for s in line.split(',')]
            thresh = int(lfields[11]) # kt
            if thresh == 0:
                continue
            attrs = [f'R{thresh}_{quad}' for quad in ('NE','SE','SW','NW')]
            for attr in attrs:
                setattr(self, attr, int(lfields[BDeck.attr_indices[attr]]))
        # Set wind radii that are not present to missing
        radii_attrs = [f'R{thresh}_{q}' for thresh in (34, 50, 64) for q in ('NE','SE','SW','NW')]
        for attr in radii_attrs:
            if not getattr(self, attr, None):
                setattr(self, attr, np.nan)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return f'<BDeckEntry storm={self.stormID} time={self.time:%Y%m%d%H}>'


class Storm:
    """
    Storm object constructed from ATCF B-deck data, containing time-ordered
    arrays of storm attributes

    Attributes
    ----------
        ID:             [str]             Storm ID (e.g., '01L')
        name:           [str]             Storm name ('NONAME' if no defined name)
        basin:          [array(str)]      2-letter basin IDs at each time (e.g., 'AL')
        classification: [array(str)]      2-letter storm classification (e.g., 'TS')
        lat:            [array(float)]    Latitude (deg N)
        lon:            [array(float)]    Longitude ([-180, 180])
        time:           [array(datetime)] Observation time
        vmax:           [array(float)]    Maximum sustained wind (kt)
        maxgust:        [array(float)]    Maximum wind gusts (kt)
        pmin:           [array(float)]    Minimum central pressure (hPa)
        rmw:            [array(float)]    Radius of maximum wind (nm)
        roci:           [array(float)]    Radius of outermost closed isobar (nm)
        poci:           [array(float)]    Pressure of outermost closed isobar (hPa)
        eye_diameter:   [array(float)]    Eye diameter (nm)
        R34_{quad}:     [array(float)]    Radius of 34kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R50_{quad}:     [array(float)]    Radius of 50kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
        R64_{quad}:     [array(float)]    Radius of 64kt wind (nm) in the `quad` quadrant,
                                          where `quad` is one of ['NE','SE','SW','NW']
    """
    def __init__(self, bdeck):
        """
        Args
        ----
            bdeck: BDeck instance
        """
        # Collect time-dependent attributes
        dtypes = {
            'basin': 'U2', 'lat': float, 'lon': float, 'time': 'datetime64[m]',
            'vmax': float, 'pmin': float, 'rmw': float, 'poci': float, 'roci': float,
            'maxgust': float, 'eye_diameter': float, 'R34_NE': float, 'R34_SE': float,
            'R34_SW': float, 'R34_NW': float, 'R50_NE': float, 'R50_SE': float,
            'R50_SW': float, 'R50_NW': float, 'R64_NE': float, 'R64_SE': float,
            'R64_SW': float, 'R64_NW': float, 'classification': 'U2'
        }
        # Create arrays
        for attr, dtype in dtypes.items():
            arr = np.array([getattr(entry, attr) for entry in bdeck.entries], dtype=dtype)
            setattr(self, attr, arr)

        # Collect time-invariant attributes. Make sure to use latest entry
        self.ID = bdeck.entries[-1].stormID
        self.name = bdeck.entries[-1].stormname

    def __hash__(self):
        return hash((self.ID, self.name))
