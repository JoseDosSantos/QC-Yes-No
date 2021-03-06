import csv

def main(path='data/'):
    with open(path + 'all_pre_pos.txt', 'w', newline='') as a:
        reader = csv.reader(open(path + 'all_cleaned.csv', 'r'), delimiter=';')
        for i, line in enumerate(reader):
            for word in [str(i)] + (line[0].split('?')[0].split()) + ['?']:
                a.write(word + '\n')
            a.write('\n')
        a.flush()
        a.close()

    with open(path + 'labels.csv', 'w', newline='') as a:
        writer = csv.writer(a, delimiter=';')
        reader = csv.reader(open(path + 'all_cleaned.csv', 'r'), delimiter=';')
        for i, line in enumerate(reader):
            writer.writerow([i, line[1]])
        a.flush()
        a.close()

if __name__ == '__main__':
    main()