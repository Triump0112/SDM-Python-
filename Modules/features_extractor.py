from shapely.geometry import Point, shape, Polygon
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# Bioclimatic variable names
bioclim_names = {
    'bio01': 'annual_mean_temperature',
    'bio02': 'mean_diurnal_range',
    'bio03': 'isothermality',
    'bio04': 'temperature_seasonality',
    'bio05': 'max_temperature_warmest_month',
    'bio06': 'min_temperature_coldest_month',
    'bio07': 'temperature_annual_range',
    'bio08': 'mean_temperature_wettest_quarter',
    'bio09': 'mean_temperature_driest_quarter',
    'bio10': 'mean_temperature_warmest_quarter',
    'bio11': 'mean_temperature_coldest_quarter',
    'bio12': 'annual_precipitation',
    'bio13': 'precipitation_wettest_month',
    'bio14': 'precipitation_driest_month',
    'bio15': 'precipitation_seasonality',
    'bio16': 'precipitation_wettest_quarter',
    'bio17': 'precipitation_driest_quarter',
    'bio18': 'precipitation_warmest_quarter',
    'bio19': 'precipitation_coldest_quarter'
}

class Feature_Extractor:
    def __init__(self, ee):
        self.ee = ee
        self.assets = self.load_assets()
        self.min_max_values = self.get_region_min_max_features()

    def load_assets(self):
        ee = self.ee
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017').filter(ee.Filter.eq('country_co', 'IN'))
        bioclim = ee.Image("WORLDCLIM/V1/BIO")
        malabar_ecoregion = ecoregions.filterBounds(india).filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')).first()
        species_occurrences = ee.FeatureCollection("projects/sigma-bay-425614-a6/assets/Mangifera_Malabar_f")
        lulc = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1").filterBounds(malabar_ecoregion.geometry()).select('label')
        modeLULC = lulc.mode().clip(malabar_ecoregion.geometry())

        additional_images = {
            'annual_precipitation': ee.Image("projects/ee-plantationsitescores/assets/AnnualPrecipitation"),
            'aridity_index': ee.Image("projects/ee-plantationsitescores/assets/India-AridityIndex"),
            'topsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_PH_H2O"),
            'subsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_PH_H2O"),
            'topsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_TEXTURE"),
            'subsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_USDA_TEX_CLASS"),
            'elevation': ee.Image("USGS/SRTMGL1_003").select('elevation')
        }

        return {
            'bioclim': bioclim,
            'malabar_ecoregion': malabar_ecoregion,
            'species_occurrences': species_occurrences,
            'modeLULC': modeLULC,
            **additional_images
        }

    def get_feature_values_at_point(self, lat, lon):
        point = self.ee.Geometry.Point(lon, lat)
        all_values = {}
        try:
            bioclim_values = self.assets['bioclim'].reduceRegion(
                reducer=self.ee.Reducer.first(),
                geometry=point,
                scale=30,
                maxPixels=1e13
            ).getInfo()

            for bio_code, bio_name in bioclim_names.items():
                all_values[bio_name] = bioclim_values.get(bio_code, float('nan'))

            image_assets = {
                'aridity_index': self.assets['aridity_index'],
                'topsoil_ph': self.assets['topsoil_ph'],
                'subsoil_ph': self.assets['subsoil_ph'],
                'topsoil_texture': self.assets['topsoil_texture'],
                'subsoil_texture': self.assets['subsoil_texture'],
                'elevation': self.assets['elevation']
            }

            for name, asset in image_assets.items():
                value = asset.reduceRegion(
                    reducer=self.ee.Reducer.first(),
                    geometry=point,
                    scale=10,
                    maxPixels=1e13
                ).get('b1' if name != 'elevation' else 'elevation').getInfo()
                all_values[name] = value if value is not None else float('nan')

        except Exception as e:
            print(f"Error in get_feature_values_at_point: {str(e)}")
            return None

        return all_values

    def get_region_min_max_features(self):
        region = self.assets['malabar_ecoregion'].geometry()
        min_max_dict = {}

        for bio_code, bio_name in bioclim_names.items():
            try:
                band = self.assets['bioclim'].select([bio_code])
                min_val = band.reduceRegion(
                    reducer=self.ee.Reducer.min(),
                    geometry=region,
                    scale=500,
                    maxPixels=1e13
                ).getInfo().get(bio_code, float('nan'))
                max_val = band.reduceRegion(
                    reducer=self.ee.Reducer.max(),
                    geometry=region,
                    scale=500,
                    maxPixels=1e13
                ).getInfo().get(bio_code, float('nan'))
                min_max_dict[bio_name] = {'min': min_val, 'max': max_val}
            except Exception as e:
                print(f"Error getting min/max for {bio_name} ({bio_code}): {str(e)}")
                min_max_dict[bio_name] = {'min': float('nan'), 'max': float('nan')}

        return min_max_dict

    def normalize_bioclim_values(self, values_dict):
        normalized = {}
        for key, value in values_dict.items():
            if value is None:
                normalized[key] = None
                continue
            if key in self.min_max_values:
                min_val = self.min_max_values[key]['min']
                max_val = self.min_max_values[key]['max']
                if max_val - min_val == 0:
                    normalized[key] = 0
                else:
                    normalized[key] = (value - min_val) / (max_val - min_val)
        return normalized

    def process_point(self, row):
        lat, lon = row['latitude'], row['longitude']
        if pd.isna(lat) or pd.isna(lon):
            return None
        values = self.get_feature_values_at_point(lat, lon)
        if values:
            return {'longitude': lon, 'latitude': lat, **self.normalize_bioclim_values(values)}
        return None

    def add_features(self, occurrences, batch_size=4000):
        total_size = occurrences.shape[0]
        num_batches = (total_size + batch_size - 1) // batch_size
        all_presence_points = []

        batches = [
            occurrences.iloc[i * batch_size: min((i + 1) * batch_size, total_size)]
            for i in range(num_batches)
        ]

        with ProcessPoolExecutor() as executor:
            future_to_batch = {executor.submit(self.process_batch, batch): batch for batch in batches}
            for future in as_completed(future_to_batch):
                try:
                    all_presence_points.extend(future.result())
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")

        return pd.DataFrame(all_presence_points)

    def process_batch(self, batch_df):
        results = []
        for _, row in batch_df.iterrows():
            try:
                result = self.process_point(row)
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing point: {str(e)}")
        return results
