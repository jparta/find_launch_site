import time
from typing import cast

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from shapely.geometry.base import BaseGeometry

from bounds import main_crs, metric_crs
from tools import get_distance


class Glob:
    def __init__(self, geometry: BaseGeometry | list[BaseGeometry], ids: list[str], tags: dict[str, str]):
        self.geometry: list[BaseGeometry] = geometry if isinstance(geometry, list) else [geometry]
        self.ids: list[str] = ids
        self.tags: dict[str, str] = tags
    
    def merge_with(self, other: "Glob"):
        self.geometry.extend(other.geometry)
        self.ids.extend(other.ids)
        self.tags = {**self.tags, **other.tags}

    def to_dict_with_unioned_geometry(self):
        return {
            "geometry": shapely.unary_union(self.geometry),
            "ids": self.ids,
            "tags": self.tags,
        }


# geometry by full_id
def get_geometry_by_full_id(geodf: gpd.GeoDataFrame, full_id: str):
    geometries = geodf[geodf["full_id"] == full_id].geometry
    if len(geometries) > 1:
        raise ValueError(f"More than one geometry for full_id {full_id}")
    return geometries.iloc[0]

# %% [markdown]
# ### Unify what can be unified

def glob_unification_with_tags(gdf):
    # %%
    sjoin_start_time = time.perf_counter()
    gdf_sjoined = gdf.sjoin(gdf, how="left", op="overlaps")
    print(f"sjoin time: {time.perf_counter() - sjoin_start_time}")

    gdf_sjoin_overlapping_ids = set(list(gdf_sjoined["full_id_left"]) + list(gdf_sjoined["full_id_right"]))
    # print(f"sjoined ids: {gdf_sjoin_overlapping_ids}")

    geometry_dict_creation_start_time = time.perf_counter()
    geometry_by_full_id = {
        row.full_id: row.geometry for index_val, row in gdf.iterrows()
    }
    print(f"geometry_dict_creation time: {time.perf_counter() - geometry_dict_creation_start_time}")

    # %%
    # Do sjoin to find what is intersecting, then do manual merge of those geometries and tags
    do_report = False
    reporting_interval = len(gdf) // 25

    # "globs", i.e. polygons which intersect with each other
    # each glob is a row
    # each id of a feature has a glob it belongs to
    id_to_glob: dict[str, Glob] = {}
    merge_times_s = []
    glob_to_merge_creation_times_s = []
    iteration_times_s = []
    id_update_times_s = []
    creating_new_glob_times_s = []
    getting_geometry_by_full_id_from_dict_times_s = []
    row_namedtuples_creation_start_time = time.perf_counter()
    row_namedtuples = gdf_sjoined.itertuples()
    print(f"row_namedtuples_creation time: {time.perf_counter() - row_namedtuples_creation_start_time}")
    for i, row_namedtuple in enumerate(gdf_sjoined.itertuples()):
        iteration_start_time = time.perf_counter()
        if do_report and i != 0 and i % reporting_interval == 0:
            print(f"{i} / {len(gdf_sjoined)}")
            print(f"iteration average: {np.mean(iteration_times_s) * 1e6:.1f} us, max: {np.max(iteration_times_s) * 1e6:.1f} us, min: {np.min(iteration_times_s) * 1e6:.1f} us, std: {np.std(iteration_times_s) * 1e6:.1f} us")
            print(f"glob to merge creation average : {np.mean(glob_to_merge_creation_times_s) * 1e6:.1f} us, {np.mean(glob_to_merge_creation_times_s) / np.mean(iteration_times_s) * 100:.1f} % of total, max: {np.max(glob_to_merge_creation_times_s) * 1e6:.1f} us, min: {np.min(glob_to_merge_creation_times_s) * 1e6:.1f} us, std: {np.std(glob_to_merge_creation_times_s) * 1e6:.1f} us")
            print(f"merge average : {np.mean(merge_times_s) * 1e6:.1f} us, {np.mean(merge_times_s) / np.mean(iteration_times_s) * 100:.1f} % of total, max: {np.max(merge_times_s) * 1e6:.1f} us, min: {np.min(merge_times_s) * 1e6:.1f} us, std: {np.std(merge_times_s) * 1e6:.1f} us")
            print(f"id update average: {np.mean(id_update_times_s) * 1e6:.1f} us, % time spent updating id: {np.mean(id_update_times_s) / np.mean(iteration_times_s) * 100:.1f} % of total, max: {np.max(id_update_times_s) * 1e6:.1f} us, min: {np.min(id_update_times_s) * 1e6:.1f} us, std: {np.std(id_update_times_s) * 1e6:.1f} us")
            print(f"creating new glob average: {np.mean(creating_new_glob_times_s) * 1e6:.1f} us, {np.mean(creating_new_glob_times_s) / np.mean(iteration_times_s) * 100:.1f} % of total, max: {np.max(creating_new_glob_times_s) * 1e6:.1f} us, min: {np.min(creating_new_glob_times_s) * 1e6:.1f} us, std: {np.std(creating_new_glob_times_s) * 1e6:.1f} us")
            print(f"getting geometry by full id from dict average : {np.mean(getting_geometry_by_full_id_from_dict_times_s) * 1e6:.1f} us, {np.mean(getting_geometry_by_full_id_from_dict_times_s) / np.mean(iteration_times_s) * 100:.1f} % of total, max: {np.max(getting_geometry_by_full_id_from_dict_times_s) * 1e6:.1f} us, min: {np.min(getting_geometry_by_full_id_from_dict_times_s) * 1e6:.1f} us, std: {np.std(getting_geometry_by_full_id_from_dict_times_s) * 1e6:.1f} us")
        id_left = row_namedtuple.full_id_left
        id_right = row_namedtuple.full_id_right
        if id_left == id_right:
            # skip self-intersections
            continue
        if id_left in id_to_glob and not id_right in id_to_glob:
            # merge right into left
            id_of_existing_glob = id_left
            id_to_merge = id_right
            tags_to_merge = row_namedtuple.tags_right
        elif id_right in id_to_glob and not id_left in id_to_glob:
            # merge left into right
            id_of_existing_glob = id_right
            id_to_merge = id_left
            tags_to_merge = row_namedtuple.tags_left
        elif id_left in id_to_glob and id_right in id_to_glob:
            # both are in a glob already, so skip
            continue
        else:
            # neither are in a glob, so create a new one
            creating_new_glob_start = time.perf_counter()
            getting_geometry_by_full_id_from_dict_start = time.perf_counter()
            new_glob_geometry = geometry_by_full_id[id_left]
            getting_geometry_by_full_id_from_dict_times_s.append(time.perf_counter() - getting_geometry_by_full_id_from_dict_start)
            new_glob = Glob(
                new_glob_geometry,
                [id_left],
                row_namedtuple.tags_left,
            )
            creating_new_glob_times_s.append(time.perf_counter() - creating_new_glob_start)
            id_to_glob[id_left] = new_glob
            if id_right is np.nan:
                # left doesn't intersect with anything, so skip merging
                continue
            id_of_existing_glob = id_left
            id_to_merge = id_right
            tags_to_merge = row_namedtuple.tags_right
        existing_glob = id_to_glob[id_of_existing_glob]
        glob_to_merge_creation_start = time.perf_counter()
        geometry_to_merge = geometry_by_full_id[id_to_merge]
        glob_to_merge = Glob(
            geometry_to_merge,
            [id_to_merge],
            tags_to_merge,
        )
        glob_to_merge_creation_times_s.append(time.perf_counter() - glob_to_merge_creation_start)
        merge_start = time.perf_counter()
        existing_glob.merge_with(glob_to_merge)
        merge_times_s.append(time.perf_counter() - merge_start)
        merged_glob = existing_glob
        # update id_to_glob
        id_update_start = time.perf_counter()
        for id_to_update in merged_glob.ids:
            id_to_glob[id_to_update] = merged_glob
        id_update_times_s.append(time.perf_counter() - id_update_start)
        iteration_times_s.append(time.perf_counter() - iteration_start_time)


    # %%
    # Find features tagges with "aeroway"

    gdf.apply(get_distance, axis=1).value_counts()

    # %%
    globs = set(id_to_glob.values())

    # %%
    print(f"count of polys before merging: {len(gdf)}")
    print(f"count of polys after merging: {len(globs)}")

    # %% [markdown]
    # ### Reconsitute dataframe

    # %%
    glob_records = (glob.to_dict_with_unioned_geometry() for glob in globs)
    df_from_sjoined = pd.DataFrame.from_records(glob_records)
    # Add back in the features that didn't overlap with anything
    gdf_did_not_overlap = gdf[~gdf.full_id.isin(gdf_sjoin_overlapping_ids)].drop(columns=["full_id"])
    concatenated = pd.concat([gdf_did_not_overlap, df_from_sjoined], ignore_index=True)
    gdf = gpd.GeoDataFrame(concatenated, geometry="geometry", crs=main_crs)

    gdf = gdf.explode()
    return gdf


def simplify_geodataframe(gdf):
    gdf = gdf.to_crs(metric_crs)
    gdf = cast(gpd.GeoDataFrame, gdf)
    simplification_tolerance_meters = 10
    gdf.geometry = gdf.simplify(simplification_tolerance_meters)
    return gdf
