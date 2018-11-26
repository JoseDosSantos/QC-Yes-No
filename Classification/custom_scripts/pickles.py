import pickle
DEFAULT_PICKLE_LOCATION = 'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\pickles'

def create_pickle(data, name):
    with open(DEFAULT_PICKLE_LOCATION + name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=2)

def load_pickle(name):
    with open(DEFAULT_PICKLE_LOCATION + name + '.pickle', 'rb') as f:
        return pickle.load(f)