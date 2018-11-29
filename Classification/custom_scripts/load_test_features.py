import sys
sys.path.append('..\\preprocessing')
import util
import config as cfg
import pipeline
import combine_sources
import import_data
import pickles
import test_fs
import pandas as pd
from collections import Counter
from nltk.util import ngrams

def get_labels():
    features = util.load_pickle(name='fs_0.9', path='..\\pickles\\feature_sets\\')
    print(Counter(features.data_set['Label']))

def create_dataset(source, target):
    files = combine_sources.main(source, target)
    new_files = {'all': files}
    pipeline.main(new_files, target)

def run(sets, reload_dataset=False):
    if reload_dataset:
        source = {'eval': cfg.EVAL}
        target = 'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Corpus\\Evaluation\\'
        create_dataset(source, target)
    test_data = import_data.load_data('C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Corpus\\Evaluation\\data_ready.csv')
    #sets = [0.1]#, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]
    for s in sets:
        fs = util.load_pickle(name='fs_' + str(s), path='..\\pickles\\feature_sets\\')
        t = test_fs.TestFS(test_data)
        t.generate_features(fs)
        t.export(name=str(s), path='..\\pickles\\test_features\\fs_test_')
        print('Exported test for {}'.format(str(s)))


if __name__ == '__main__':
    sets = [1]
    run(sets)


