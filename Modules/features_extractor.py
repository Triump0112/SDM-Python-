from shapely.geometry import Point, shape, Polygon
import pandas as pd 
import concurrent.futures

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

class Feature_Extractor():
    def __init__(self,ee):
        self.ee = ee 
        self.assets = self.load_assets()
        self.min_max_values = self.get_region_min_max_features()



    def load_assets(self):
        # Load the ecoregions dataset
        ee = self.ee 
        ecoregions = ee.FeatureCollection('RESOLVE/ECOREGIONS/2017')

        # Load the India boundaries dataset
        india = ee.FeatureCollection('USDOS/LSIB_SIMPLE/2017') \
            .filter(ee.Filter.eq('country_co', 'IN'))

        # Load the bioclimatic data (WorldClim)
        bioclim = ee.Image("WORLDCLIM/V1/BIO")

        # Filter the ecoregions to get the Malabar Coast moist forests
        malabar_ecoregion = ecoregions \
            .filterBounds(india) \
            .filter(ee.Filter.eq('ECO_NAME', 'Malabar Coast moist forests')) \
            .first()

        # Load the species occurrences dataset
        species_occurrences = ee.FeatureCollection("projects/sigma-bay-425614-a6/assets/Mangifera_Malabar_f")

        # Get the LULC (Land Use/Land Cover) data for the eco region
        lulc = ee.ImageCollection("GOOGLE/DYNAMICWORLD/V1") \
            .filterBounds(malabar_ecoregion.geometry()) \
            .select('label')  # We only need the 'label' band for LULC classification

        # Get the most common LULC class (mode) for the Malabar region
        modeLULC = lulc.mode().clip(malabar_ecoregion.geometry())

        # Load additional datasets
        additional_images = {
            'annual_precipitation': ee.Image("projects/ee-plantationsitescores/assets/AnnualPrecipitation"),
            'aridity_index': ee.Image("projects/ee-plantationsitescores/assets/India-AridityIndex"),
            'topsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_PH_H2O"),
            'subsoil_ph': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_PH_H2O"),
            'topsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-T_TEXTURE"),
            'subsoil_texture': ee.Image("projects/ee-plantationsitescores/assets/Raster-S_USDA_TEX_CLASS"),
            'elevation': ee.Image("USGS/SRTMGL1_003").select('elevation')  # Added elevation dataset
        }

        # Create a dictionary of all loaded assets
        loaded_assets = {
            'bioclim': bioclim,
            'malabar_ecoregion': malabar_ecoregion,
            'species_occurrences': species_occurrences,
            'modeLULC': modeLULC,
            **additional_images  # Unpacking the additional images into the main dictionary
        }

        return loaded_assets


    def get_feature_values_at_point(self,lat, lon):
        assets = self.assets 
        ee = self.ee 
        point = ee.Geometry.Point(lon, lat)
        all_values = {}

        # Bioclim variable mapping


        try:
            # Get bioclim values
        
            bioclim_values = assets['bioclim'].reduceRegion(
                reducer=ee.Reducer.first(),
                geometry=point,
                scale=30,
                maxPixels=1e13
            ).getInfo()

            # Add bioclim values with descriptive names
            for bio_code, bio_name in bioclim_names.items():
                all_values[bio_name] = bioclim_values.get(bio_code, float('nan'))

            # Get values for other image assets
            image_assets = {
                'aridity_index': assets['aridity_index'],
                'topsoil_ph': assets['topsoil_ph'],
                'subsoil_ph': assets['subsoil_ph'],
                'topsoil_texture': assets['topsoil_texture'],
                'subsoil_texture': assets['subsoil_texture'],
                'elevation': assets['elevation']
            }

            for name, asset in image_assets.items():
                try:
                    if name == 'elevation':
                        value = asset.reduceRegion(
                            reducer=ee.Reducer.first(),
                            geometry=point,
                            scale=10,
                            maxPixels=1e13
                        ).get('elevation').getInfo()
                    else:
                        value = asset.reduceRegion(
                            reducer=ee.Reducer.first(),
                            geometry=point,
                            scale=10,
                            maxPixels=1e13
                        ).get('b1').getInfo()
                    all_values[name] = value if value is not None else float('nan')
                except Exception as e:
                    print(f"Error getting {name} value: {str(e)}")
                    all_values[name] = float('nan')


        except Exception as e:
            print(f"Error in get_feature_values_at_point: {str(e)}")
            return None

        return all_values

    def get_region_min_max_features(self):
        assets = self.assets
        region = assets['malabar_ecoregion']
        region = region.geometry()
        ee= self.ee 
        try:
            bioclim_region = assets['bioclim']
        except Exception as e:
            print(f"Error getting bioclim region: {str(e)}")
            return None


        min_max_dict = {}


        for bio_code, bio_name in bioclim_names.items():
            try:
                # Select the specific bioclim band
                band = bioclim_region.select([bio_code])

                # Calculate min value
                min_info = band.reduceRegion(
                    reducer=ee.Reducer.min(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e13
                ).getInfo()
                min_value = min_info.get(bio_code, float('nan'))

                # Calculate max value
                max_info = band.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e13
                ).getInfo()
                max_value = max_info.get(bio_code, float('nan'))

                # Store min and max values
                min_max_dict[bio_name] = {'min': min_value, 'max': max_value}
            except Exception as e:
                print(f"Error getting min/max for {bio_name} ({bio_code}): {str(e)}")
                min_max_dict[bio_name] = {'min': float('nan'), 'max': float('nan')}

        # Get min/max values for additional features
        image_assets = {
            'aridity_index': assets['aridity_index'],
            'topsoil_ph': assets['topsoil_ph'],
            'subsoil_ph': assets['subsoil_ph'],
            'topsoil_texture': assets['topsoil_texture'],
            'subsoil_texture': assets['subsoil_texture'],
            'elevation': assets['elevation']
        }

        for asset_name, asset in image_assets.items():
            try:
                # Calculate min value for the asset
                min_info = asset.reduceRegion(
                    reducer=ee.Reducer.min(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e13
                ).getInfo()
                min_value = min_info.get('b1' if asset_name != 'elevation' else 'elevation', float('nan'))

                # Calculate max value for the asset
                max_info = asset.reduceRegion(
                    reducer=ee.Reducer.max(),
                    geometry=region,
                    scale=1000,
                    maxPixels=1e13
                ).getInfo()
                max_value = max_info.get('b1' if asset_name != 'elevation' else 'elevation', float('nan'))

                # Store min and max values
                min_max_dict[asset_name] = {'min': min_value, 'max': max_value}
            except Exception as e:
                print(f"Error getting min/max for {asset_name}: {str(e)}")
                min_max_dict[asset_name] = {'min': float('nan'), 'max': float('nan')}

        return min_max_dict
    
    def normalize_bioclim_values(self,values_dict):
        min_max_dict = self.min_max_values
        normalized = {}
        cnt=0
        Norm_Max = 1
        for key, value in values_dict.items():
            if value is None:
                cnt+=1

                normalized[key] = None  # or you can set a default value like 0
                continue

            if type(value) not in [float, int]:
                print(f"Invalid value type for {key}: {value}. Expected float or int.")
                continue

            if key in min_max_dict:
                min_val = min_max_dict[key]['min']
                max_val = min_max_dict[key]['max']
                if max_val - min_val == 0:
                    normalized[key] = 0
                else:
                    normalized[key] = 1*((value - min_val)*Norm_Max) / (max_val - min_val)
        if cnt>0:print(f"Warning: None value for {cnt}. Skipping normalization for these many keys.")
        return normalized
    
    def process_point(self,row):
        assets, min_max_dict = self.assets, self.min_max_values
        latitude = row['latitude']
        longitude = row['longitude']

        # Check if latitude or longitude is NaN, skip if so
        if pd.isna(latitude) or pd.isna(longitude):
            return None

        values = self.get_feature_values_at_point(latitude, longitude)
        normalized_values = self.normalize_bioclim_values(values)
        return {'longitude': longitude, 'latitude': latitude, **normalized_values}
    


    def add_features(self,occurrences,batch_size=4000):

        total_size = occurrences.shape[0]
        num_batches = (total_size + batch_size - 1) // batch_size
        all_presence_points = []

        for i in range(num_batches):
            start_idx, end_idx = i * batch_size, min((i + 1) * batch_size, total_size)
            batch_df = occurrences.iloc[start_idx:end_idx]

            # Use ThreadPoolExecutor to process points in parallel
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = [executor.submit(self.process_point, row) for _, row in batch_df.iterrows()]
                for future in concurrent.futures.as_completed(futures):
                    result = future.result()
                    if result is not None:
                        all_presence_points.append(result)
                    # print(f'{len(all_presence_points)} done')  # Print progress

        return pd.DataFrame(all_presence_points)
