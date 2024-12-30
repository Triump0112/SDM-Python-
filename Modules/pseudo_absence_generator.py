import numpy as np
import pandas as pd
from shapely.geometry import Point, shape
from tqdm.notebook import tqdm
import concurrent.futures
from scipy.spatial.distance import cdist
from . import features_extractor
from . import LULC_filter

feature_cols = [
    'annual_mean_temperature',
    'mean_diurnal_range',
    'isothermality',
    'temperature_seasonality',
    'max_temperature_warmest_month',
    'min_temperature_coldest_month',
    'temperature_annual_range',
    'mean_temperature_wettest_quarter',
    'mean_temperature_driest_quarter',
    'mean_temperature_warmest_quarter',
    'mean_temperature_coldest_quarter',
    'annual_precipitation',
    'precipitation_wettest_month',
    'precipitation_driest_month',
    'precipitation_seasonality',
    'precipitation_wettest_quarter',
    'precipitation_driest_quarter',
    'precipitation_warmest_quarter',
    'precipitation_coldest_quarter',
    'aridity_index',
    'topsoil_ph',
    'subsoil_ph',
    'topsoil_texture',
    'subsoil_texture',
    'elevation'
]

class PseudoAbsences:
    def __init__(self, ee):
        self.ee = ee
        self.feature_extractor = features_extractor.Feature_Extractor(self.ee)
        self.modeLULC = LULC_filter.LULC_Filter(self.ee).load_modeLULC()
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'IN'))
        ecoregion_geom = ecoregions.filterBounds(india).filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')).first()
        self.ecoregion_geom = ecoregion_geom.geometry()

    def reliability(self, presence_points_df, absence_point_dict):
        presence_features = presence_points_df.iloc[:, 2:]
        absence_point_features = [float(absence_point_dict[i]) for i in absence_point_dict if absence_point_dict[i] is not None]
        distances = cdist([absence_point_features], presence_features, metric='euclidean')
        similarities = np.exp(-distances**2 / (2 * presence_features.shape[1]))
        mean_similarity = np.nanmean(similarities)
        reliability = 1 - mean_similarity
        return reliability

    def generate_batch_points(self, minx, miny, maxx, maxy, batch_size=1000):
        rand_lons = np.random.uniform(minx, maxx, batch_size)
        rand_lats = np.random.uniform(miny, maxy, batch_size)
        return list(zip(rand_lons, rand_lats))

    def generate_pseudo_absences(self, presence_df):
        num_points = len(presence_df)
        ecoregion_geom = self.ecoregion_geom
        eco_region_polygon = shape(ecoregion_geom.getInfo())
        bounds = eco_region_polygon.bounds
        minx, miny, maxx, maxy = bounds
        global_df = pd.DataFrame(columns=['longitude', 'latitude', 'reliability', 'normalized_features'])
        total_attempts = 0
        batch_size = 1000

        with tqdm(total=num_points, desc="Generating points") as pbar:
            while len(global_df) < num_points:
                batch_points = self.generate_batch_points(minx, miny, maxx, maxy, batch_size)
                total_attempts += batch_size

                with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
                    future_to_point = {
                        executor.submit(self.process_single_point, lon, lat, eco_region_polygon, presence_df): (lon, lat)
                        for lon, lat in batch_points
                    }

                    for future in concurrent.futures.as_completed(future_to_point):
                        result = future.result()
                        if result is not None:
                            global_df = pd.concat([global_df, result], ignore_index=True)
                            pbar.update(1)

                        if len(global_df) >= num_points:
                            break

        print(f"\nFinal Statistics:")
        print(f"Total points generated: {len(global_df)}")
        print(f"Total attempts made: {total_attempts}")
        print(f"Overall success rate: {(len(global_df) / total_attempts) * 100:.2f}%")

        for i, col in enumerate(feature_cols):
            global_df[col] = global_df['normalized_features'].apply(lambda x: x[col])
        global_df = global_df.drop('normalized_features', axis=1)

        return global_df

    def process_single_point(self, rand_lon, rand_lat, eco_region_polygon, presence_df):
        try:
            if eco_region_polygon.contains(Point(rand_lon, rand_lat)):
                point = self.ee.Geometry.Point([rand_lon, rand_lat])
                lulc_value = self.modeLULC.reduceRegion(
                    self.ee.Reducer.mode(),
                    point,
                    scale=10,
                    maxPixels=1e9
                ).get('label').getInfo()

                if lulc_value == 1:
                    bioclim_values_random = self.feature_extractor.get_feature_values_at_point(rand_lat, rand_lon)
                    normalized_bioclim_values_random = self.feature_extractor.normalize_bioclim_values(bioclim_values_random)
                    reliability_value = self.reliability(presence_df, normalized_bioclim_values_random)

                    if reliability_value > 0.04:
                        row = {
                            'longitude': rand_lon,
                            'latitude': rand_lat,
                            'reliability': reliability_value,
                            'normalized_features': normalized_bioclim_values_random
                        }
                        return pd.DataFrame([row])
        except Exception:
            pass
        return None
