from feature_set_generator import FeatureSetGenerator
from import_data import load_data
from collections import Counter
from time import time
import load_test_features

data_set = load_data()
print('Data loaded.')
sets = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
for i in sets:
    s = time()
    f = FeatureSetGenerator(data_set=data_set, size=i, full_init=True, load_from_file=False, export=True)
    print('Saved featureset with {0}% of data. Time Taken:{1}'.format(i*100, time()-s))
    print(Counter(f.data_set['Label']))

load_test_features.run()

