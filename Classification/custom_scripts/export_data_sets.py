from feature_set_generator import FeatureSetGenerator
from import_data import load_data
from collections import Counter
from time import time

def run(sets):
    data_set = load_data()
    print('Data loaded.')
    for i in sets:
        s = time()
        f = FeatureSetGenerator(data_set=data_set, size=i, full_init=True, load_from_file=False, export=True)
        print('Saved featureset with {0}% of data. Time Taken:{1}'.format(i*100, time()-s))
        print(Counter(f.data_set['Label']))



