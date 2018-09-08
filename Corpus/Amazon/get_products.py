from bs4 import BeautifulSoup
import simple_get
import re
import csv
import time
import os

CATEGORIES = ['Elektronik & Computer', 'Haushalt, Garten, Baumarkt', 'Beauty, Drogerie & Lebensmittel', 'Spielzeug & Baby', 'Kleidung, Schuhe & Uhren', 'Sport & Freizeit',
              'Auto, Motorrad & Gewerbe']



def get_product_link(url):
    pattern_numbers_strings = re.compile(r'\/[A-Z0-9]{10}')
    pattern_numbers = re.compile(r'[0-9]+')
    pattern_strings = re.compile(r'[A-Z]+')
    match = pattern_numbers_strings.findall(url)
    if match:
        if pattern_numbers.findall(match[0]):
            if pattern_strings.findall(match[0]):
                return match[0]
    return None


def get_pages(url_list):
    for link_id, link in enumerate(url_list):
        name = link.rsplit('/')[1]
        product_ids = []
        if os.path.exists('C:\\Users\\Josef\\PycharmProjects\\QC-Yes-No\\Tweet extraction\\products\\' + name + '.csv'):
            print(name, ' already found')
        else:
            try:
                try:
                    full_url = 'https://www.amazon.de' + link
                    page = BeautifulSoup(simple_get.simple_get(full_url), 'html.parser')
                except:
                    print('None found')
                    continue
                products = page.find_all('a')
                for i, paragraph in enumerate(products):
                    if paragraph.has_attr('href'):
                        id = get_product_link(paragraph['href'])
                        if id:
                            if not id in product_ids:
                                product_ids.append(id)
                with open(name + '.csv', 'w', newline='') as output_file:
                    writer = csv.writer(output_file, delimiter=';')
                    for id in set(product_ids):
                        writer.writerow([id])
                    output_file.flush()
                time.sleep(2)
            except Exception as e:
                print(e)


def load_category_links():
    raw_html = simple_get.simple_get('https://amazon.de/gp/site-directory/ref=nav_shopall_btn')
    html = BeautifulSoup(raw_html, 'html.parser')
    category_table = html.find('table',  {"id": "shopAllLinks"})
    big_category_div = category_table.find_all('div', class_='popover-grouping')
    all_categories = []
    for i, div in enumerate(big_category_div):
        if div.find('h2').text in CATEGORIES:
            for link in div.find_all('a'):
                if 'node' in link['href']:
                    all_categories.append(link['href'])
    get_pages(all_categories)


load_category_links()
