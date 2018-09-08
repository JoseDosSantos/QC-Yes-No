from nltk.parse.stanford import StanfordDependencyParser

path_to_jar = 'stanford/stanford-corenlp-3.9.1.jar'
path_to_models_jar = 'stanford-german-corenlp-2018-02-27-models.jar'

dependency_parser = StanfordDependencyParser(path_to_jar=path_to_jar, path_to_models_jar=path_to_models_jar)