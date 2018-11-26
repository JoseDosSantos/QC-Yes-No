import re
import csv

def clean(string):
    #return string
    clean_string = string.replace(u'\xa0', u' ')
    clean_string = re.sub(r'\d+', 'Zahl', clean_string)
    clean_string = re.sub(r'\n', ' ', clean_string)
    clean_string = re.sub(r'°', ' Grad ', clean_string)
    clean_string = re.sub(r'24/7', 'immer', clean_string)
    clean_string = re.sub(r'/', ' ', clean_string)
    clean_string = re.sub(r'%', ' Prozent ', clean_string)
    clean_string = re.sub(r'[Zz]\.?[Bb]\.?\s', 'zum Beispiel', clean_string)
    clean_string = re.sub(r'[Dd]\.?[Hh]\.?\s', 'das heißt', clean_string)
    clean_string = re.sub(r'[Bb][Ss][Pp][Ww]\.?\s', 'beispielsweise', clean_string)
    clean_string = re.sub(r'[Uu][Ss][Ww]\.?\s', 'und so weiter', clean_string)
    clean_string = re.sub(r'[Hh]allo\s', '', clean_string)
    clean_string = re.sub(r'[Hh]i\s', '', clean_string)
    clean_string = re.sub(r'[Hh]ey\s', '', clean_string)
    clean_string = re.sub(r'[Gg]uten\s[Mm]orgen\s', '', clean_string)
    clean_string = re.sub(r'[Gg]uten\s[Aa]bend\s', '', clean_string)
    clean_string = re.sub(r'[Gg]uten\s[Tt]ag\s', '', clean_string)
#todo: 's zu es
    clean_string = re.sub(r'(\([^)]*\))', ' ', clean_string)
    clean_string = re.sub(r'"', '', clean_string)
    clean_string = re.sub(r'\+', '', clean_string)
    clean_string = re.sub(r'-', '', clean_string)
    clean_string = re.sub(r',', ' ', clean_string)
    clean_string = re.sub(r'\^', '', clean_string)
    clean_string = re.sub(r'\'', '', clean_string)
    clean_string = re.sub(r'`', '', clean_string)
    clean_string = re.sub(r'´', '', clean_string)
    clean_string = re.sub(r'#', '', clean_string)
    clean_string = re.sub(r':', '', clean_string)
    clean_string = re.sub(r'!', '', clean_string)

    clean_string = re.sub(r'\'', '', clean_string)
    clean_string = re.sub(r'\.', '', clean_string)
    clean_string = re.sub(r'\s{2,}', ' ', clean_string)
    clean_string = re.sub(r'\s(?=\?)', ' ', clean_string)
    clean_string = re.sub(r'\?*(?=(?:\?))', '', clean_string)
    #clean_string = re.sub(r'\?', '', clean_string)
    clean_string = clean_string.strip()
    #bitte, danke, und, eigentlich, überhaupt, git, wirklich
    return clean_string#.lower()


def main():
    with open('data/' + 'all_cleaned.csv', 'w', newline='') as a:
        writer = csv.writer(a, delimiter=';')
        reader = csv.reader(open('data/' + 'all.csv', 'r'), delimiter=';')
        for line in reader:
            writer.writerow([clean(line[0]), line[1]])
        a.flush()
        a.close()

if __name__ == '__main__':
    main()