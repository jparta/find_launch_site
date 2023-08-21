import pyproj

main_crs = "EPSG:4326"

espoo_kirkkonummi_bbox = (60.05, 24.45, 60.25, 24.90)
uusimaa_bbox = (59.75, 23.75, 60.75, 25.75)
bbox_of_view_bounds = espoo_kirkkonummi_bbox

middle_of_view_bounds_longitude = (
    bbox_of_view_bounds[1] + bbox_of_view_bounds[3]
) / 2
metric_crs = f"+proj=tmerc +lon_0={middle_of_view_bounds_longitude} +x_0=500000 +k=0.9996"

transform_main_to_metric = pyproj.Transformer.from_proj(
    main_crs,
    metric_crs,
)

bbox_xy = (
    bbox_of_view_bounds[1],
    bbox_of_view_bounds[0],
    bbox_of_view_bounds[3],
    bbox_of_view_bounds[2],
)
bbox_xx_yy = (
    bbox_of_view_bounds[1],
    bbox_of_view_bounds[3],
    bbox_of_view_bounds[0],
    bbox_of_view_bounds[2],
)
bbox_coords = map(str, bbox_of_view_bounds)
bbox_string = ",".join(bbox_coords)
