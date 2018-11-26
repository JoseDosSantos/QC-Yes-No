import csv
import sys

def append_files(files, name):
    with open('data/' + name + '.csv', 'w', newline='') as a:
        writer = csv.writer(a, delimiter=';')
        for file in files:
            reader = csv.reader(open(file, 'r'), delimiter=';')
            for line in reader:
                writer.writerow([line[0], line[1]])
        a.flush()
        a.close()

def main(sets):
    for i in sets:
        append_files(sets[i], i)

if __name__ == '__main__':
    main(sys.argv[1:])