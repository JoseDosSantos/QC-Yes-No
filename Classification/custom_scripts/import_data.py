import pandas as pd
import csv


DATA_PATH = "C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Classification\\preprocessing\\data\\data_ready.csv"

# For reading in csv files
def load_data(path=DATA_PATH):
    data_set = pd.read_csv(path, sep=';').drop('Unnamed: 0', 1)
    data_set['Feature'] = data_set['Feature'].apply(lambda x: x.split('#'))
    data_set['PosTags'] = data_set['PosTags'].apply(lambda x: x.split('#'))
    return data_set

# Export the separated data_set, help_function
def save_data(data_set):
    with open('raw.csv', 'w', encoding='UTF-8', newline='') as f:
        writer = csv.writer(f)
        for _, line in data_set.iterrows():
            a = ' '.join(line['Feature'][:-1])
            writer.writerow([a, line['Label']])
        f.flush()
        f.close()