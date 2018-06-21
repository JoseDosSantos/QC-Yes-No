import nltk

from nltk.tokenize import sent_tokenize
nltk.data.path.append('C:\\Users\\jcersovsky\\PycharmProjects\\QC\\nltk_data')

text = "Veränderungen über einen Walzer. Veränderungen über einen Walzer, Veränderungen über einen Walzer. Veränderungen über einen Walzer"

sent_tokenize_list = sent_tokenize(text)
pass