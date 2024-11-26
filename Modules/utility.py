from . import features_extractor 
from shapely.geometry import Point, Polygon, box
from shapely.wkt import loads
import random
import pandas as pd 


def divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=10):
    polygon = loads(polygon)
    min_x, min_y, max_x, max_y = polygon.bounds
    step_x = (max_x - min_x) / grid_size
    step_y = (max_y - min_y) / grid_size
    sampled_points = []
    total= 0
    for i in range(grid_size):
        for j in range(grid_size):
            # Define the current grid cell as a polygon
            cell_min_x = min_x + i * step_x
            cell_max_x = min_x + (i + 1) * step_x
            cell_min_y = min_y + j * step_y
            cell_max_y = min_y + (j + 1) * step_y
            grid_cell = box(cell_min_x, cell_min_y, cell_max_x, cell_max_y)

            # Get the intersection of the grid cell with the ecoregion
            intersection = polygon.intersection(grid_cell)

            if not intersection.is_empty:
                # Sample points within the intersection
                points_in_cell = []
                while len(points_in_cell) < points_per_cell:
                    # Generate a random point within the grid cell bounds
                    random_x = random.uniform(cell_min_x, cell_max_x)
                    random_y = random.uniform(cell_min_y, cell_max_y)
                    point = Point(random_x, random_y)
                    # Check if the point lies within the intersection
                    if intersection.contains(point):
                        points_in_cell.append(point)
                total+=len(points_in_cell)
                # print(total)
                sampled_points.extend(points_in_cell)

    # Convert points to a list of [longitude, latitude] pairs
    # print('points sampled',total)
    sampled_points = [[point.x, point.y] for point in sampled_points]

    # Now create the DataFrame with correct shape (longitude, latitude)
    sampled_points = pd.DataFrame(sampled_points, columns=["longitude", "latitude"])

    return sampled_points


def representative_feature_vector_for_polygon(sampled_points, ee):

    feature_Extractor = features_extractor.Feature_Extractor(ee)
    

    features_df = feature_Extractor.add_features(sampled_points)
    

    feature_vector = features_df.mean(axis=0, skipna=True).tolist()
    feature_vector=feature_vector[2:]
    return feature_vector






def create_similarity_matrix():
    SM=[[0 for _ in range(49)] for _ in range(49)]
    return SM 


