import requests 
import csv 
from time import sleep 
import ee 
import pandas as pd

class Presence_dataloader():
    def __init__(self):
        # self.ee = ee
        return 
    
    def load_raw_presence_data(self, maxp=2000):
        with open("Inputs/polygon.wkt", "r") as input_polygon:
            polygon_wkt = input_polygon.read().strip()  # Fix: Add parentheses to call strip()

        with open("Inputs/genus_name.txt", "r") as genus:
            genus_name = genus.read().strip()  # Read genus name properly
        
        occurrence_points = set()

        try:
            with open("data/presence.csv", "w") as presence_data:
                writer = csv.writer(presence_data)
                writer.writerow(["longitude", "latitude"])
        except FileNotFoundError:
            pass 

        offset, limit = 0, 300  # API pagination parameters
        
        print(f'Beginning to find at least {maxp} presence points for {genus_name} in input polygon')
        
        while True:
            gbif_url = "https://api.gbif.org/v1/occurrence/search"
            params = {
                "scientificName": genus_name + "%",
                "geometry": polygon_wkt,
                "limit": limit,
                "offset": offset,
                "eventDate": "2017-01-01,2023-12-31",
                "kingdomKey": 6
            }
            response = requests.get(gbif_url, params=params)
            response.raise_for_status()

            new_points = set()
            for result in response.json()["results"]:
                point = (result["decimalLongitude"], result["decimalLatitude"])
                if point not in occurrence_points:
                    new_points.add(point)
                    occurrence_points.add(point)
                    print('genus is', result['genus'], 'species is', result['scientificName'])

            if new_points:
                with open("data/presence.csv", "a", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerows(new_points)
                print(f"Saved {len(new_points)} new unique occurrence points to presence.csv")

            if len(response.json()["results"]) < limit:
                break

            offset += limit

            if len(occurrence_points) >= maxp:  # Fix: Correct the comparison condition
                break 

        return occurrence_points
    
    def load_unique_lon_lats(self):

        df = pd.read_csv("data/presence.csv")
    
        df_unique = df.drop_duplicates(subset=['longitude', 'latitude'])
        
        return df_unique
    
