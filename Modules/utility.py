from . import features_extractor 
from shapely.geometry import Point, Polygon, box
from shapely.wkt import loads
import random
import pandas as pd 
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
import os 


def divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=5):
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

def find_representive_vectors_from_files(input_folder, ee):
    feature_vectors = []
    file_names = []
    
    # Iterate over all WKT files in the given folder
    for filename in os.listdir(input_folder):
        if filename.endswith('.wkt'):
            print('processing',filename)
            with open(os.path.join(input_folder, filename), 'r') as file:
                polygon_wkt = file.read().strip()
                
            # Ensure the polygon_wkt is in string format (WKT format), not a Polygon object
            if isinstance(polygon_wkt, str):
                # Now, correctly load the WKT string into a Polygon object
                polygon = polygon_wkt
                
                # Generate sampled points from the polygon
                sampled_points = divide_polygon_to_grids(polygon, grid_size=10, points_per_cell=10)
                
                # Generate the representative feature vector for the polygon
                feature_vector = representative_feature_vector_for_polygon(sampled_points, ee)
                
                # Store the vector and corresponding file name (for identification)
                feature_vectors.append(feature_vector)
                file_names.append(filename)
            else:
                print(f"Skipping {filename} because it is not a valid WKT string.")
    
    # Convert feature vectors into a DataFrame
    feature_vectors_df = pd.DataFrame(feature_vectors)
    feature_vectors_df.index = file_names  # Use file names as the index
    feature_vectors_df.to_csv('data/representative_vectors_eco_region_wise.csv')
    
    return feature_vectors_df


def calculate_cosine_similarity_matrix(feature_vectors_df):
    similarity_matrix = cosine_similarity(feature_vectors_df)
    return similarity_matrix

# Function to calculate Euclidean Distance similarity matrix
def calculate_euclidean_similarity_matrix(feature_vectors_df):
    # Euclidean distance is often interpreted as the inverse of the distance (the smaller the distance, the higher the similarity)
    distance_matrix = euclidean_distances(feature_vectors_df)
    similarity_matrix = 1 / (1 + distance_matrix)  # Add 1 to avoid division by zero and make it a similarity measure
    return similarity_matrix






def save_matrix_to_text(matrix, filename, labels):
    """
    Save matrix to a human-readable text file with row and column labels.
    
    Parameters:
    matrix (np.ndarray): Similarity matrix to save
    filename (str): Output filename
    labels (list): Row and column labels
    """
    with open(filename, 'w') as f:
        # Write column headers
        f.write(' ' * 50)  # Indent for row labels
        f.write('\t'.join(labels) + '\n')
        
        # Write matrix with row labels
        for i, row_label in enumerate(labels):
            # Format row with row label and values
            row_values = [f"{val:.4f}" for val in matrix[i]]
            formatted_row = f"{row_label:<50}\t" + '\t'.join(row_values) + '\n'
            f.write(formatted_row)


