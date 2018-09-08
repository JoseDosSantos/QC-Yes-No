import simple_get
from bs4 import BeautifulSoup
import csv
import os
from random import shuffle
from copy import deepcopy

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def load_files():
    products = []
    path = 'C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Tweet extraction\\products\\'
    for file in os.listdir(path):
        with open(path + file, 'r') as csvfile:
            reader = csv.reader(csvfile, delimiter=';')
            for row in reader:
                products.append(row[0])
    p_set = set(products)
    p_list = list(p_set)
    shuffle(p_list)
    generator = chunks(p_list, 50)
    print('Iterating', str(int(round(len(p_list) / 50))), 'parts of 50 products.')

    for i, products in enumerate(generator):
        questions = []
        for id in products:
            try:
                questions += get_questions(id)
            except:
                pass
        with open('questions\\products_part_' + str(i) + '.csv', 'w', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=';')
            for line in questions:
                try:
                    writer.writerow([line])
                except:
                    pass
            output_file.flush()
            print('Finished batch ', i)


def get_questions(id):
    try:
        questions = []
        question_page = 'https://amazon.de/ask/questions/asin/' + id
        page = BeautifulSoup(simple_get.simple_get(question_page), 'html.parser')
        for question in page.select('div[id*="question"]')[:5]:
            text = question.find('a').text.strip()
            if not text.endswith('?'):
                text += '?'
            questions.append(text)
        return questions
    except:
        return None

load_files()