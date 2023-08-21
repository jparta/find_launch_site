from concurrent.futures import as_completed
from pprint import pformat
from textwrap import indent

import osm2geojson
from pyproj.enums import TransformDirection
import requests
from requests_futures.sessions import FuturesSession
from shapely import Point

from bad_launch import OSMTagWithMinDistance, keep_away_from_when_launching
from bounds import bbox_of_view_bounds, transform_main_to_metric


overpass_url = "http://overpass-api.de/api/interpreter"

union_query_template = """[out:json][timeout:25];
(
{union_members}
);
out body;
>;
out skel qt;
"""

not_working_well_union_query_template = """[out:json][timeout:25];
(
{union_members}
);
(._;>;);
convert item ::=::,::geom=geom(),_osm_type=type();
out skel qt;
"""

everything_query_template = """[out:json][timeout:25];
(
  way({bbox});
  relation({bbox});
);
out body;
>;
out skel qt;
"""


all_polygons_at_point_query = """[out:json][timeout:25];
is_in({lat_lon});
out body;
>;
out skel qt;
"""





# %%
def osm_tags_to_query(tags: list[OSMTagWithMinDistance]) -> str:
    feature_types = ["node", "way", "relation"]
    simple_query_template = "{feature_type}[{tag}]({bbox});"
    union_members = []
    for feature_type in feature_types:
        for tag in tags:
            format_map = {
                "feature_type": feature_type,
                "tag": tag,
                "bbox": "{bbox}",
            }
            simple_query = simple_query_template.format(**format_map)
            union_members.append(simple_query)
    union_members_on_lines = "\n".join(union_members)
    union_members_indented = indent(union_members_on_lines, "  ")
    prepared_query = union_query_template.format(union_members=union_members_indented)
    return prepared_query


def make_queries():
    # Create queries grouped by distance
    distances = set([tag.distance for tag in keep_away_from_when_launching])
    bad_stuff_tags_by_distance = {
        distance: [tag for tag in keep_away_from_when_launching if tag.distance == distance]
        for distance in distances
    }

    queries_without_bboxes = {
        distance: osm_tags_to_query(tags)
        for distance, tags in bad_stuff_tags_by_distance.items()
    }

    def expand_bbox(
        bbox: tuple[float, float, float, float], distance: int
    ) -> tuple[float, float, float, float]:
        """Expand the bbox by the distance + a bit extra to include possible bad stuff which
        affects the possible launch area from far away (large distance value).
        """
        padding_multiplier = 1.01
        padded_distance = distance * padding_multiplier
        expanded_bbox = (
            bbox[0] - padded_distance,
            bbox[1] - padded_distance,
            bbox[2] + padded_distance,
            bbox[3] + padded_distance,
        )
        return expanded_bbox

    queries_with_bboxes: dict[int, str] = {}
    for distance, query in queries_without_bboxes.items():
        # Expand the bbox by the distance to include possible bad stuff with a
        # large distance which should be taken into account.
        print(f"for distance: {distance}")
        # print(f"before expand_bbox: {bbox_of_view_bounds}")

        # To metric for the expand_bbox function
        bbox_of_view_metric = transform_main_to_metric.transform_bounds(*bbox_of_view_bounds)
        # print(f"bbox_of_view_metric: {bbox_of_view_metric}")

        expanded_bbox_metric = expand_bbox(bbox_of_view_metric, distance)
        # print(f"expanded_bbox_metric: {expanded_bbox_metric}")

        # Back to main crs
        expanded_bbox = transform_main_to_metric.transform_bounds(
            *expanded_bbox_metric, direction=TransformDirection.INVERSE
        )

        print(f"after expand_bbox: {expanded_bbox}")
        expanded_bbox_coords = map(str, expanded_bbox)
        expanded_bbox_string = ",".join(expanded_bbox_coords)
        query_with_bbox = query.replace("{bbox}", expanded_bbox_string)
        queries_with_bboxes[distance] = query_with_bbox
    # pprint(queries_with_bboxes)
    return queries_with_bboxes


def get_geojson_from_OSM(distance_to_query: dict[int, str]):
    # Send queries to Overpass API
    def response_hook(resp, *args, **kwargs):
        print(f"got response with url: {resp.url}")
        print(f"response status: {resp.status_code}")

        if resp.status_code != 200:
            # print(f"Error: {resp.text}")
            print(f"request body: {resp.request.body}")
            raise Exception("Error")

        result = osm2geojson.json2geojson(resp.text)
        resp.geojson = result

    session = FuturesSession()
    session.hooks["response"] = response_hook

    futures = []
    for distance, query in distance_to_query.items():
        print(f"launching query with distance {distance}")
        future = session.post(overpass_url + f"#{distance}", data=query)
        future.distance = distance
        futures.append(future)

    geojson_features = []
    for future in as_completed(futures):
        resp = future.result()
        distance = future.distance
        geojson = resp.geojson
        features = geojson["features"]
        print(f"got response for distance {distance}, {len(features)} features")
        geojson_features.extend(features)

    print(f"got {len(geojson_features)} features in total")
    return geojson_features



def point_on_OSM_polygon_not_within_bad_boundary(point_in_main_crs: Point) -> bool:
    boundary_tags = [
        ("amenity", "school"),
        ("barrier", "fence"),
        ("boundary", "administrative"),
        ("boundary", "historic"),
        ("boundary", "maritime"),
        ("boundary", "political"),
        ("boundary", "postal_code"),
        ("boundary", "public_transport"),
        ("boundary", "place"),
        ("boundary", "protected_area"),
        ("natural", "peninsula"),
        ("natural", "wetland"),
        ("landuse", "farmyard"),
        ("landuse", "brownfield"),
        ("landuse", "residential"),
        ("waterway", "boatyard"),
        ("leisure", "marina"),
        ("leisure", "pitch"),
        ("leisure", "sports_centre"),
        ("tourism", "hotel"),
    ]
    boundary_keys = [
        "airspace",
        "building",
        "was:building",
    ]
    #print(f"boundary_tags: {boundary_tags}")
    #print(f"boundary_keys: {boundary_keys}")
    print(point_in_main_crs)
    query = all_polygons_at_point_query.format(lat_lon=f"{point_in_main_crs.x},{point_in_main_crs.y}")
    #print(query)
    response = requests.post(overpass_url, data=query)
    #print(response.text)
    data = response.json()
    # pprint(data)
    elements_which_are_not_boundaries = []
    for element in data["elements"]:
        element_tags = element.get("tags")
        #print(f"element_tags: {pformat(element_tags)}")
        if element_tags is None:
            continue
        offending_tags = [
            tag
            for tag in element_tags.items()
            if tag in boundary_tags
        ]
        if offending_tags:
            # print(f"found offending tags: {pformat(offending_tags)}")
            continue
        offending_keys = [
            key
            for key in element_tags
            if key in boundary_keys
        ]
        if offending_keys:
            # print(f"found offending keys: {pformat(offending_keys)}")
            continue
        elements_which_are_not_boundaries.append(element)
    if elements_which_are_not_boundaries:
        print(f"elements_which_are_not_boundaries: {pformat([elem['tags'] for elem in elements_which_are_not_boundaries])}")
    return len(elements_which_are_not_boundaries) > 0
