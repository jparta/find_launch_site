# %%
import time
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path
from pprint import pprint, pformat

import cartopy.crs as ccrs
import cartopy.io.img_tiles as cimgt
import geofeather
import geopandas as gpd
import numpy as np
import pandas as pd
import pyproj
from matplotlib import pyplot as plt
from pyproj.enums import TransformDirection
from shapely import Polygon, Point, LineString
from shapely.ops import polylabel, transform
from shapely.validation import explain_validity
from typing import cast

from bad_launch import (
    key_val_to_tag_string,
    keep_away_from_when_launching,
)
from bounds import main_crs, metric_crs, bbox_xy, bbox_xx_yy, bbox_of_view_bounds, transform_main_to_metric
from geometry_optimization import glob_unification_with_tags, simplify_geodataframe
from query_osm import make_queries, get_geojson_from_OSM, point_on_OSM_polygon_not_within_bad_boundary
from tools import get_distance, linestring_to_polygon, count_geom_vertices

# %%
# TODO

#  - Add to version control
#  - Instead using "areas to avoid" in the initial exclusion step, exclude them when checking candidates
#  - Check why places like 60.140580 24.668884 are given as suggestions even though they should be already excluded (in this case, "landuse:residential")




# Form GeoDataFrame, filter columns
def geojson_features_to_geodataframe(geojson_features):
    gdf = gpd.GeoDataFrame.from_features(geojson_features, crs=main_crs)
    gdf = cast(gpd.GeoDataFrame, gdf)
    return gdf


def deduplicate_based_on_full_id(gdf):
    before_dedup = len(gdf)
    gdf = gdf.drop_duplicates(subset=["full_id"])
    after_dedup = len(gdf)
    print(f"Removed {before_dedup - after_dedup} duplicates based on full_id")
    return gdf


def clean_up_data_and_limit_columns(gdf):
    pprint(list(gdf.columns))

    gdf["full_id"] = gdf["type"] + "/" + gdf["id"].astype(str)

    columns_to_keep = ["geometry", "full_id", "tags"]
    gdf = gdf[columns_to_keep]
    gdf = cast(gpd.GeoDataFrame, gdf)

    gdf = deduplicate_based_on_full_id(gdf)
    return gdf


def linestrings_to_polygons(gdf):
    linestrings_before_polygonize = gdf[gdf.apply(lambda row: row.geometry.type == "LineString", axis=1)].copy()

    gdf.geometry = gdf.geometry.apply(linestring_to_polygon)

    linestrings_after_polygonize = gdf[gdf.apply(lambda row: row.geometry.type == "LineString", axis=1)].copy()
    not_linestrings_anymore = linestrings_before_polygonize - linestrings_after_polygonize

    print(not_linestrings_anymore)

    print(gdf.geometry.type.value_counts())
    return gdf


def report_vertices_count(gdf):
    counts = gdf.geometry.apply(count_geom_vertices)
    total_vertices = counts.sum()
    print(f"total vertices: {total_vertices}")


def prepare_bad_launching_areas_and_features():
    queries = make_queries()
    geojson_features = get_geojson_from_OSM(queries)
    gdf = geojson_features_to_geodataframe(geojson_features)

    gdf = clean_up_data_and_limit_columns(gdf)
    gdf = linestrings_to_polygons(gdf)

    # gdf = glob_unification_with_tags(gdf)

    gdf = cast(gpd.GeoDataFrame, gdf)
    gdf = gdf.explode()

    gdf.geometry = gdf.make_valid()

    gdf = simplify_geodataframe(gdf)

    report_vertices_count(gdf)

    return gdf


# Buffer and find suitable areas (not yet necessarily on land)
# With multiprocessing
def distance_buffering_operation(distance: float, to_buffer: gpd.GeoDataFrame) -> gpd.GeoSeries:
    """It is assumed that the distance is in metric crs
    """
    simplify_tolerance_meters = 10
    # Make sure all distances are the same
    if not np.isclose(to_buffer["distance"], distance, rtol=0.01, atol=0.1).all():
        raise ValueError("All distances must be the same")
    gdf_buffered = to_buffer.buffer(distance)
    gdf_buffered = cast(gpd.GeoSeries, gdf_buffered)
    gdf_buffered = gdf_buffered.simplify(simplify_tolerance_meters)
    return gdf_buffered


def split_given_size(a, size):
    return np.split(a, np.arange(size,len(a),size))


def buffer_based_on_desired_distance(gdf: gpd.GeoDataFrame) -> gpd.GeoSeries:
    # Buffer each distance in a separate process
    distance_finding_start = time.perf_counter()
    gdf["distance"] = gdf.apply(get_distance, axis=1)
    print(gdf["distance"].value_counts())
    print(f"distance finding took {time.perf_counter() - distance_finding_start:.1f} s")
    # reproject to meters for adding buffer
    gdf_buffering = gdf.to_crs(metric_crs)
    gdf_buffering = cast(gpd.GeoDataFrame, gdf_buffering)
    # Geometry fix

    buffering_start = time.perf_counter()
    grouped_by_distance = [(distance, rows) for distance, rows in gdf_buffering.groupby("distance")]
    lengths = {distance: len(rows) for distance, rows in grouped_by_distance}
    print("lengths", pformat(lengths))
    grouped_by_distance_and_split = []
    for distance, rows in grouped_by_distance:
        partition_max_size = 50
        rows_split = split_given_size(rows, partition_max_size)
        grouped_by_distance_and_split.extend([(distance, rows) for rows in rows_split])
    print("length of grouped_by_distance_and_split", len(grouped_by_distance_and_split))
    # print([type(g) for g in grouped_by_distance])
    multiprocessing_start = time.perf_counter()
    with Pool() as pool:
        gdf_buffered_list = pool.starmap(distance_buffering_operation, grouped_by_distance_and_split)
    print(f"multiprocessing took {time.perf_counter() - multiprocessing_start:.1f} s")
    concat_start = time.perf_counter()
    gdf_buffered = gpd.GeoSeries( pd.concat(gdf_buffered_list), crs=metric_crs )
    print(f"concat took {time.perf_counter() - concat_start:.1f} s")
    gdf_buffered = gdf_buffered.to_crs(main_crs)
    print(f"buffering took {time.perf_counter() - buffering_start:.1f} s")
    return gdf_buffered


def get_suitable_launch_area(unsuitable_area: gpd.GeoSeries) -> gpd.GeoSeries:
    difference_start = time.perf_counter()
    bbox_polygon_xy = Polygon.from_bounds(*bbox_xy)
    possible_launch_area_from_difference = bbox_polygon_xy.difference(
        unsuitable_area.unary_union
    )
    print(f"difference took {time.perf_counter() - difference_start:.1f} s")
    total_vertices = count_geom_vertices(possible_launch_area_from_difference)
    print(f"total vertices: {total_vertices}")
    possible_launch_area = gpd.GeoSeries(possible_launch_area_from_difference, crs=main_crs)
    return possible_launch_area


def filter_suitable_polygons_by_surface_area(geoseries: gpd.GeoSeries) -> gpd.GeoSeries:
    minimum_polygon_area_meters = 100

    print(f"before explode: {len(geoseries)}")
    exploded_possible_launch_area = geoseries.explode(ignore_index=True)
    print(f"after explode: {len(exploded_possible_launch_area)}")
    possible_launch_area_metric = exploded_possible_launch_area.to_crs(metric_crs)
    possible_launch_area = exploded_possible_launch_area[
        possible_launch_area_metric.area >= minimum_polygon_area_meters
    ]
    print(f"after filtering by area: {len(possible_launch_area)}")
    print(
        f"minimum area after filtering: {possible_launch_area.to_crs(metric_crs).area.min()}"
    )
    return possible_launch_area


def limit_suitable_areas_to_land(suitable_areas_before_limiting_to_land: gpd.GeoSeries):
    # intersect with land, to find suitable area that is not in the sea
    start = time.time()

    land_geofeather_filepath = Path("data") / "land-polygons-split-3857" / "land_polygons.feather"

    land_polygons_from_coastline_loaded = geofeather.from_geofeather(
        land_geofeather_filepath
    )
    print(f"loading land_polygons_from_coastline took {time.time() - start:.1f} s")

    land_polygons_from_coastline_crs = land_polygons_from_coastline_loaded.crs
    #print(f"crs of land_polygons_from_coastline: {land_polygons_from_coastline_crs}")

    viewbox_limited_land_polygons_intersection_start = time.time()
    project_source_to_coastline = pyproj.Transformer.from_proj(
        main_crs,
        land_polygons_from_coastline_crs,
    )

    bbox_of_view_polygon = Polygon.from_bounds(*bbox_of_view_bounds)
    transformed_bbox_polygon = transform(
        project_source_to_coastline.transform, bbox_of_view_polygon
    )

    # Print bounds
    print("bbox_of_view_polygon bounds:", bbox_of_view_polygon.bounds)
    print("transformed_bbox_polygon bounds:", transformed_bbox_polygon.bounds)
    print(
        f"land_polygons_from_coastline bounds: {land_polygons_from_coastline_loaded.total_bounds}"
    )

    land_polygons_from_coastline_in_viewbox = land_polygons_from_coastline_loaded.intersection(
        transformed_bbox_polygon
    )
    print("Found land polygons intersecting bbox", time.time() - start)
    print(f"number of land_polygons_from_coastline: {len(land_polygons_from_coastline_in_viewbox)}")
    print(
        f"land_polygons_from_coastline bounds: {land_polygons_from_coastline_in_viewbox.total_bounds}"
    )

    land_polygons_from_coastline_in_viewbox = land_polygons_from_coastline_in_viewbox.to_crs(main_crs)
    print(f"Reprojected land polygons to {main_crs}", time.time() - start)

    # Do union of all land polygons
    land_polygons_from_coastline_union = land_polygons_from_coastline_in_viewbox.unary_union
    # land_from_coastline = gpd.GeoSeries(land_polygons_from_coastline_union, crs=main_crs)
    print("Unioned land polygons", time.time() - start)

    suitable_area_on_land = suitable_areas_before_limiting_to_land.intersection(
        land_polygons_from_coastline_union
    ).explode(index_parts=False)
    print(f"intersection with land using viewbox limited took {time.time() - viewbox_limited_land_polygons_intersection_start:.1f} s")

    print("Found possible launch area on land", time.time() - start)
    return suitable_area_on_land


def get_suitable_launch_area_on_land(gdf):
    gdf_buffered = buffer_based_on_desired_distance(gdf)
    possible_launch_area = get_suitable_launch_area(gdf_buffered)
    possible_launch_area = filter_suitable_polygons_by_surface_area(possible_launch_area)
    suitable_launch_area_on_land = limit_suitable_areas_to_land(possible_launch_area)
    return suitable_launch_area_on_land


def test_against_known_sites(suitable_launch_area_on_land):
    is_nice_area_for_launch_testset = [
        (Point(60.217595, 24.600559), False),
        (Point(60.189062, 24.8200870), False),
        (Point(60.152483, 24.719278), False),
        (Point(60.154029, 24.775810), False),
    ]

    is_nice_area_for_launch_testset = [
        (Point(p.y, p.x), is_land_expected)
        for p, is_land_expected in is_nice_area_for_launch_testset
    ]

    land_checking_gdf = suitable_launch_area_on_land.copy().to_crs(main_crs)
    for point, is_land_expected in is_nice_area_for_launch_testset:
        is_land_estimated = land_checking_gdf.contains(point).any()
        correct_estimation = is_land_estimated == is_land_expected
        # Using emoji to make it easier to spot
        print(
            f"{point} is nice place to launch: {is_land_estimated} {'✅' if correct_estimation else '❌'}"
        )


# Remove all but the largest polygons
def filter_to_largest_polygons(gdf):
    top_n = 150
    largest_polygons_launch_area: gpd.GeoSeries = gdf.to_crs(
        metric_crs
    )
    print(
        f"areas of largest polygons: {largest_polygons_launch_area.area.sort_values(ascending=False).head(top_n)}"
    )
    index_of_largest = list(
        largest_polygons_launch_area.area.sort_values(ascending=False).head(top_n).index
    )
    largest_polygons_shapely = sorted(largest_polygons_launch_area, key=lambda p: p.area)[-top_n:]

    largest_polygons_gdf = gpd.GeoDataFrame(
        geometry=largest_polygons_shapely, crs=metric_crs
    )
    return largest_polygons_gdf


def find_suggested_launch_locations(possible_launch_area_gdf: gpd.GeoDataFrame, n_suggestions: int):
    # %%
    """We'll have to check each polygon separately because polylabel doesn't support MultiPolygons.
    Candidates are points, one per polygon at a time. The top candidate is the one which is
    farthest away from the polygon's boundary. The top candidate is added to suggestions and the
    circle is removed from the polygon. This is repeated until we have enough suggestions. Each
    time a top candidate (new suggestion) is chosen, the polygon where the suggestion was found in
    is added to polygons_to_check. The other polygons cannot contain a better candidate than the
    ones we have already found, so they are not checked again.
    """

    from copy import deepcopy
    from dataclasses import dataclass


    @dataclass(frozen=True)
    class PointAndDistance:
        point: Point
        distance_from_poly: float


    def candidate_point_to_main_crs_point(candidate_point: Point) -> Point:
        return Point(
            transform_main_to_metric.transform(*candidate_point.xy, direction=TransformDirection.INVERSE)
        )

    def candidate_is_viable(candidate: PointAndDistance) -> bool:
        return point_on_OSM_polygon_not_within_bad_boundary(candidate_point_to_main_crs_point(candidate.point))

    def remove_candidate(candidate: PointAndDistance) -> None:
        """Remove from candidate list, remove circle from polygon, add poly back to pool of polygons to check"""
        circle_relation_to_distance_from_poly = 0.5
        poly = candidate_to_poly.pop(candidate)
        circle_to_remove_radius = min(size_of_removed_circle_meters, candidate.distance_from_poly * circle_relation_to_distance_from_poly)
        circle_to_remove = candidate.point.buffer(size_of_removed_circle_meters)
        poly = poly.difference(circle_to_remove)
        if poly.type == "Polygon":
            polygons_to_check.append(poly)
        elif poly.type == "MultiPolygon":
            polygons_to_check.extend(poly.geoms)

    polylabel_tolerance = 5
    size_of_removed_circle_meters = 10
    reporting_interval = 1

    suggestions: list[Point] = []
    # These were rejected because they were on areas which are not suitable for launching.
    rejected_candidates: list[Point] = []
    # Candidates are points, one per polygon at a time. They represent possible launch locations.
    # A polygon is always either in candidate_to_poly or in polygons_to_check.
    # A polygon is in polygons_to_check if a candidate has not been computed for it yet.
    # A polygon is in candidate_to_poly if a candidate has been computed for it but it has not
    # been selected as top candidate and added to suggestions yet.
    polygons_to_check = list(possible_launch_area_gdf.to_crs(metric_crs).geometry)
    candidate_to_poly: dict[PointAndDistance, Polygon] = {}
    print(f"number of polygons_to_check: {len(polygons_to_check)}")
    start = time.time()
    while len(suggestions) < n_suggestions:
        while polygons_to_check:
            poly = polygons_to_check.pop()
            candidate_point: Point = polylabel(poly, polylabel_tolerance)
            distance_to_boundary = poly.boundary.distance(candidate_point)
            candidate_with_distance = PointAndDistance(
                candidate_point, distance_to_boundary
            )
            candidate_to_poly[candidate_with_distance] = poly
        # The top candidate is the one which is farthest away from its polygon's boundary.
        top_candidate = None
        found_actual_top_candidate = False
        while not found_actual_top_candidate:
            possible_top_candidate = max(candidate_to_poly, key=lambda candidate: candidate.distance_from_poly)
            if candidate_is_viable(possible_top_candidate):
                top_candidate = possible_top_candidate
                actual_top_candidate_found = True
                break
            else:
                rejected_candidates.append(possible_top_candidate.point)
                remove_candidate(possible_top_candidate)
        if top_candidate is None:
            raise RuntimeError("No viable candidate found.")
        suggestions.append(top_candidate.point)
        remove_candidate(top_candidate)

        print(f"top candidate: {top_candidate}")



    # %%
    suggestions_series = gpd.GeoSeries(suggestions, crs=metric_crs)
    suggestions_gdf = suggestions_series.to_crs(main_crs)
    len(suggestions_gdf)

    # %%
    rejected_candidates_series = gpd.GeoSeries(rejected_candidates, crs=metric_crs)
    rejected_candidates_gdf = rejected_candidates_series.to_crs(main_crs)
    len(rejected_candidates_gdf)


    # %%
    # print Google Maps URLs for the suggestions
    def print_google_maps_url(point: Point):
        print(f"https://www.google.com/maps/search/?api=1&query={point.y},{point.x}")


    for suggestion in suggestions_gdf:
        print_google_maps_url(suggestion)

    # %%
    # print OSM URLs for the suggestions
    def print_osm_url(point: Point):
        print(f"https://www.openstreetmap.org/search?query={point.y},{point.x}")


    for suggestion in suggestions_gdf:
        print_osm_url(suggestion)

    return suggestions_gdf


def main():
    gdf = prepare_bad_launching_areas_and_features()
    suitable_area_on_land = get_suitable_launch_area_on_land(gdf)
    print(f"suitable_area_on_land: {suitable_area_on_land}")
    largest_polygons_gdf = filter_to_largest_polygons(suitable_area_on_land)
    print(f"largest_polygons_gdf: {largest_polygons_gdf}")
    suggested_launch_locations = find_suggested_launch_locations(largest_polygons_gdf, 10)
    return suggested_launch_locations


if __name__ == "__main__":
    main()
