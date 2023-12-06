

def key_val_to_tag_string(key: str, value: str | None) -> str:
    if value is None or value == "":
        return f'"{key}"'
    else:
        return f'"{key}"="{value}"'


class OSMTagWithMinDistance:
    """OSM tag with a minimum distance to the launch site.
    Distance is in meters.
    """

    def __init__(self, key, value, distance: int):
        self.key = key
        self.value = value
        self.distance = distance
        if not isinstance(self.key, str):
            raise TypeError(f"Key must be a string, not '{type(self.key)}'")

    def as_dict(self):
        return {
            "key": self.key,
            "value": self.value,
            "distance": self.distance,
        }

    def __repr__(self):
        return f"OSMTagKey({self.key}, {self.value}, {self.distance})"

    def __str__(self) -> str:
        return key_val_to_tag_string(self.key, self.value)


tall_things = [
    OSMTagWithMinDistance("power", "catenary_mast", 20),
    OSMTagWithMinDistance("power", "connection", 20),
    OSMTagWithMinDistance("power", "line", 50),
    OSMTagWithMinDistance("power", "minor_line", 25),
    OSMTagWithMinDistance("power", "pole", 25),
    OSMTagWithMinDistance("power", "portal", 35),
    OSMTagWithMinDistance("power", "substation", 10),
    OSMTagWithMinDistance("power", "terminal", 35),
    OSMTagWithMinDistance("power", "tower", 50),
    OSMTagWithMinDistance("power:generator:source", "wind", 150),
    OSMTagWithMinDistance("natural", "cliff", 25),
    OSMTagWithMinDistance("natural", "coastline", 10),
    OSMTagWithMinDistance("natural", "wood", 25),
    OSMTagWithMinDistance("landuse", "forest", 25),
    OSMTagWithMinDistance("landuse", "residential", 10),
    OSMTagWithMinDistance("landcover", "trees", 25),
]

"""
Latex table

power & catenary_mast  & 20 \\
power & connection    & 20  \\
power & line & 50           \\
power & minor_line & 25     \\
power & pole & 25           \\
power & portal & 35         \\
power & substation & 10     \\
power & terminal & 35       \\
power & tower & 50          \\
power:generator:source & wind & 150 \\
natural & cliff & 25        \\
natural & coastline & 10    \\
natural & wood & 25         \\
landuse & forest & 25       \\
landuse & residential & 10  \\
landcover & trees & 25      \\
"""

bad_area_for_launch = [
    OSMTagWithMinDistance("natural", "water", 20),
    OSMTagWithMinDistance("natural", "wetland", 10),
    OSMTagWithMinDistance("place", "islet", 5),
    OSMTagWithMinDistance("place", "island", 5),
    OSMTagWithMinDistance("highway", "motorway", 20),
    OSMTagWithMinDistance("highway", "secondary", 20),
    OSMTagWithMinDistance("highway", "tertiary", 15),
    OSMTagWithMinDistance("highway", "cycleway", 10),
    OSMTagWithMinDistance("tourism", "hotel", 5),
    OSMTagWithMinDistance("leisure", "golf_course", 5),
    OSMTagWithMinDistance("natural", "bare_rock", 5),
    OSMTagWithMinDistance("natural", "scrub", 5),
    OSMTagWithMinDistance("landuse", "commercial", 20),
    OSMTagWithMinDistance("landuse", "construction", 20),
    OSMTagWithMinDistance("landuse", "farmland", 10),
    OSMTagWithMinDistance("landuse", "industrial", 20),
    OSMTagWithMinDistance("landuse", "military", 20),
    OSMTagWithMinDistance("landuse", "meadow", 10),
    OSMTagWithMinDistance("landuse", "quarry", 20),
    OSMTagWithMinDistance("landuse", "landfill", 20),
    OSMTagWithMinDistance("landuse", "basin", 20),
    OSMTagWithMinDistance("landuse", "railway", 10),
    OSMTagWithMinDistance("landuse", "reservoir", 10),
    OSMTagWithMinDistance("landuse", "residential", 20),
    OSMTagWithMinDistance("landuse", "retail", 20),
    OSMTagWithMinDistance("landuse", "education", 20),
    OSMTagWithMinDistance("landuse", "institution", 10),
    OSMTagWithMinDistance("landuse", "religious", 10),
    OSMTagWithMinDistance("landuse", "cemetery", 20),
    OSMTagWithMinDistance("landuse", "orchard", 20),
    OSMTagWithMinDistance("landuse", "vineyard", 20),
    OSMTagWithMinDistance("landuse", "depot", 20),
    OSMTagWithMinDistance("landuse", "port", 30),
]

"""
Latex table

natural & water & 20 \\
natural & wetland & 10 \\
place & islet & 5 \\
place & island & 5 \\
highway & motorway & 20 \\
highway & secondary & 20 \\
highway & tertiary & 15 \\
highway & cycleway & 10 \\
tourism & hotel & 5 \\
leisure & golf_course & 5 \\
natural & bare_rock & 5 \\
natural & scrub & 5 \\
landuse & commercial & 20 \\
landuse & construction & 20 \\
landuse & farmland & 10 \\
landuse & industrial & 20 \\
landuse & military & 20 \\
landuse & meadow & 10 \\
landuse & quarry & 20 \\
landuse & landfill & 20 \\
landuse & basin & 20 \\
landuse & railway & 10 \\
landuse & reservoir & 10 \\
landuse & residential & 20 \\
landuse & retail & 20 \\
landuse & education & 20 \\
landuse & institution & 10 \\
landuse & religious & 10 \\
landuse & cemetery & 20 \\
landuse & orchard & 20 \\
landuse & vineyard & 20 \\
landuse & depot & 20 \\
landuse & port & 30 \\
"""

"""
Rewrite table with hlines in between, and with aligned columns

natural                & water          & 20   \\
\hline
natural                & wetland        & 10   \\
\hline
place                  & islet          & 5    \\
\hline
place                  & island         & 5    \\
\hline
highway                & motorway       & 20   \\
\hline
highway                & secondary      & 20   \\
\hline
highway                & tertiary       & 15   \\
\hline
highway                & cycleway       & 10   \\
\hline
tourism                & hotel          & 5    \\
\hline
leisure                & golf\_course   & 5    \\
\hline
natural                & bare\_rock     & 5    \\
\hline
natural                & scrub          & 5    \\
\hline
landuse                & commercial     & 20   \\
\hline
landuse                & construction   & 20   \\
\hline
landuse                & farmland       & 10   \\
\hline
landuse                & industrial     & 20   \\
\hline
landuse                & military       & 20   \\
\hline
landuse                & meadow         & 10   \\
\hline
landuse                & quarry         & 20   \\
\hline
landuse                & landfill       & 20   \\
\hline
landuse                & basin          & 20   \\
\hline
landuse                & railway        & 10   \\
\hline
landuse                & reservoir      & 10   \\
\hline
landuse                & residential    & 20   \\
\hline
landuse                & retail         & 20   \\
\hline
landuse                & education      & 20   \\
\hline
landuse                & institution    & 10   \\
\hline
landuse                & religious      & 10   \\
\hline
landuse                & cemetery       & 20   \\
\hline
landuse                & orchard        & 20   \\
\hline
landuse                & vineyard       & 20   \\
\hline
landuse                & depot          & 20   \\
\hline
landuse                & port           & 30   \\
\hline

"""

be_far_away_from = [
    OSMTagWithMinDistance("aeroway", "aerodrome", 20 * 1000),
]
"""
Latex table

aeroway                & aerodrome      & 20*1000   \\
"""

keep_away_from_when_launching = tall_things + bad_area_for_launch + be_far_away_from
