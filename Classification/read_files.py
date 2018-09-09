import csv

read_files(file_directory)
for file in config.FULL_SET:
    reader = csv.reader(open(file, 'r'), delimiter=';')
    for line in reader:
        data_set.append([line[0].lower(), line[1]])