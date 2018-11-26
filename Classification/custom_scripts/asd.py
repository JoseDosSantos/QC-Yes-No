from feature_set_generator import FeatureSetGenerator
from import_data import load_data


data_set = load_data()
print('Data loaded.')

full_features = FeatureSetGenerator(data_set=data_set, size=0.8, full_init=True, load_from_file=False, export=True)