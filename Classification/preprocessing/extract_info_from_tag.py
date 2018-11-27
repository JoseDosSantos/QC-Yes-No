import pandas as pd
import csv

def main(path='data/'):

    tags = pd.read_csv(path + 'pos_tagged_encoded.csv', sep='\t', header=None, names=['Word', 'Comp Tag', 'Tag', 'Lemma'])
    labels = pd.read_csv(path + 'labels.csv', sep=';', header=None, index_col=[0], names=['Label'])

    total_labels = len(labels)
    total_rows = len(tags)
    idx = 0
    current_row = 0
    sent = []
    while idx <= total_labels - 1:
        if int(tags.loc[current_row, 'Word']) == idx:
            current_sent = []
            pos = []
            current_row += 1
            while True:
                if current_row == total_rows - 1:
                    break
                try:
                    if int(tags.loc[current_row, 'Word']) == idx + 1:
                        break
                except:
                    if '|' in tags.loc[current_row, 'Lemma']:
                        current_sent.append(tags.loc[current_row, 'Lemma'].split('|')[0])
                    elif tags.loc[current_row, 'Lemma'] != '<unknown>' and tags.loc[current_row, 'Lemma'] != '<card>':
                        current_sent.append(tags.loc[current_row, 'Lemma'])
                    else:
                        current_sent.append(tags.loc[current_row, 'Word'])
                    pos.append(tags.loc[current_row, 'Tag'])
                    current_row += 1
            sent.append([idx, current_sent, pos])
            idx += 1

    condensed = pd.DataFrame(sent, columns=['Index', 'Feature', 'PosTags']).drop('Index', 1)
    condensed['Feature'] = condensed['Feature'].apply(lambda x: '#'.join(map(str, x)))
    condensed['PosTags'] = condensed['PosTags'].apply(lambda x: '#'.join(map(str, x)))

    condensed['Label'] = labels['Label']

    condensed.to_csv(path + 'data_ready.csv', sep=';')

if __name__ == '__main__':
    main()