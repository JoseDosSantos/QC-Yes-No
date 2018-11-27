import pickle
DEFAULT_PICKLE_LOCATION = 'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\pickles'

def create_pickle(data, name, path=DEFAULT_PICKLE_LOCATION ):
    with open(path + name + '.pickle', 'wb') as f:
        pickle.dump(data, f, protocol=2)

def load_pickle(name, path=DEFAULT_PICKLE_LOCATION):
    with open(path + name + '.pickle', 'rb') as f:
        return pickle.load(f)