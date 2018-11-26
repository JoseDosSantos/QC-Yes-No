import re

def clean(line):
    clean_string = re.sub(r'Ä', 'Ae', line)
    clean_string = re.sub(r'ä', 'ae', clean_string)
    clean_string = re.sub(r'Ö', 'Oe', clean_string)
    clean_string = re.sub(r'ö', 'oe', clean_string)
    clean_string = re.sub(r'Ü', 'Ue', clean_string)
    clean_string = re.sub(r'ü', 'ue', clean_string)
    clean_string = re.sub(r'ß', 'ss', clean_string)
    return clean_string

def main():
    with open('data/' + 'pos_tagged_encoded.csv', 'w', newline='', encoding='UTF-8') as a:
        reader = open('data/' + 'pos_tagged_raw.txt', 'r')
        for line in reader:
            a.write(clean(line))
        a.flush()
        a.close()

if __name__ == '__main__':
    main()