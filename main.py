import ee 
import pandas as pd
import numpy as np
from Modules import presence_dataloader, features_extractor, LULC_filter, pseudo_absence_generator, models, Generate_Prob
ee.Authenticate()
ee.Initialize(project='sigma-bay-425614-a6')

def main():
  
    Presence_dataloader = presence_dataloader.Presence_dataloader()
    Features_extractor = features_extractor.Feature_Extractor(ee)
    LULC_Filter = LULC_filter.LULC_Filter(ee)
    Pseudo_absence = pseudo_absence_generator.PseudoAbsences(ee)
    Models = models.Models()
    generate_prob = Generate_Prob.Generate_Prob(ee)
    
    
    # # raw_occurrences = Presence_dataloader.load_raw_presence_data()   #uncomment if want to use gbif api to generate presence points
    
    # unique_presences = Presence_dataloader.load_unique_lon_lats()
    # presences_filtered_LULC = LULC_Filter.filter_by_lulc(unique_presences)
    # presence_data_with_features  = Features_extractor.add_features(presences_filtered_LULC)
    # pseudo_absence_points_with_features = Pseudo_absence.generate_pseudo_absences(presence_data_with_features)

    X, y, coords, feature_names,sample_weights = Models.load_data()
    clf, X_test, y_test, y_pred, y_proba = Models.RandomForest(X,y)
    # prob_map, transform = generate_prob.predict_eco_region(clf)
   
   
    # print(pseudo_absence_points_with_features.head(5))
    # pseudo_absence_points_with_features.to_csv('data/pseudo_absence.csv', index=False)
    
   



if __name__ == "__main__":
    main()

