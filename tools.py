import pandas as pd

from bad_launch import keep_away_from_when_launching


def split_tag_into_key_and_value(tag_string):
    """A tag is split by the last colon into key and value
    Wildcard tags of format "<key>:*" or "<key>" have value None
    """
    if ":" in tag_string:
        key, value = tag_string.rsplit(":", maxsplit=1)
        if value == "*":
            value = None
    else:
        key = tag_string
        value = None
    return key, value


# A tag is split by the last colon into key and value
# Wildcard tag: "<key>:*" or "<key>"
def tags_match(
    key1: str,
    value1: str | None,
    key2: str,
    value2: str | None,
):
    if value1 == "*":
        value1 = None
    if value2 == "*":
        value2 = None

    if not key1 or not key2:
        ValueError("key1 and key2 must be non-empty strings")

    return key1 == key2 and (value1 is None or value2 is None or value1 == value2)


def get_distance(row: pd.Series):
    distances = []
    for k, v in row["tags"].items():
        distances_for_matches = [
            osm_tag.distance
            for osm_tag in keep_away_from_when_launching
            if tags_match(k, v, osm_tag.key, osm_tag.value)
        ]
        if k == "aeroway":
            print(f"got distances for aeroway: {distances_for_matches}")
        distances.extend(distances_for_matches)
    if distances:
        return max(distances)
    else:
        raise ValueError(f"Could not find distance for {row}")

