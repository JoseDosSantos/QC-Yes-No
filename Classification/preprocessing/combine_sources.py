import csv
import sys

def append_files(files, name, path='data/'):
    with open(path + name + '.csv', 'w', newline='') as a:
        writer = csv.writer(a, delimiter=';')
        for file in files:
            reader = csv.reader(open(file, 'r'), delimiter=';')
            for line in reader:
                writer.writerow([line[0], line[1]])
        a.flush()
        a.close()


def main(sets, path='data/'):
    files = []
    for i in sets:
        append_files(sets[i], i, path)
        files.append(path + i + '.csv')
    return files

if __name__ == '__main__':
    main(sys.argv[1:])