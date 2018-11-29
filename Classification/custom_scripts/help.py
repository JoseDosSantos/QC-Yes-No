import export_data_sets
import load_test_features
import evaluate_classifiers

#sets = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9,  1]
sets = [1]
export_data_sets.run(sets)
load_test_features.run(sets)
#evaluate_classifiers.run()
