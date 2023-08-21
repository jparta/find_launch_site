

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

be_far_away_from = [
    OSMTagWithMinDistance("aeroway", "aerodrome", 20 * 1000),
]

keep_away_from_when_launching = tall_things + bad_area_for_launch + be_far_away_from
