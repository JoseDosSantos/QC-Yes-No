from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QFileDialog, qApp, QApplication, QLabel, QLineEdit, QGridLayout, QAction)
from PyQt5.QtGui import QIntValidator, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt
import pandas as pd
import sys
import os
import csv



class Classification_interface(QMainWindow):

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        openFile = QAction('&Open', self)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open file')
        openFile.triggered.connect(self.openFile)

        exitAct = QAction('&Exit', self)
        exitAct.setShortcut('Ctrl+Q')
        exitAct.setStatusTip('Exit application')
        exitAct.triggered.connect(qApp.quit)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        fileMenu.addAction(exitAct)

        title_font = QFont('Arial', 20)
        title_font.setBold(True)
        edit_font = QFont('Arial', 14)

        title = QLabel('Yes-No Classifier Interface')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title_font)

        self.file_name = QLabel('...')
        self.file_name.setStyleSheet("font-size: 14px")

        self.file_open = QPushButton('Open file...', self)
        self.file_open.setFixedHeight(30)
        self.file_open.setStyleSheet(" font-size: 14px ")

        question_number_label = QLabel('Question ID: ')
        question_number_label.setStyleSheet("font-size: 13px")

        self.question_number = QLabel('...')
        self.question_number.setStyleSheet("font-size: 13px")

        remaining_questions_label = QLabel('Remaining questions: ')
        remaining_questions_label.setStyleSheet("font-size: 13px")

        self.remaining_questions = QLabel('...')
        self.remaining_questions.setStyleSheet("font-size: 13px")

        total_questions_label = QLabel('Total questions: ')
        total_questions_label.setStyleSheet("font-size: 13px")

        self.total_questions = QLabel('...')
        self.total_questions.setStyleSheet("font-size: 13px")


        self.question = QLineEdit()
        self.question.setFixedHeight(50)
        self.question.setFont(edit_font)

        self.yes_No_Question = QPushButton('Yes-No', self)
        self.yes_No_Question.setFixedWidth(300)
        self.yes_No_Question.setFixedHeight(50)
        self.yes_No_Question.setStyleSheet("background-color: green; font-size: 20px ")
        self.yes_No_Question.setEnabled(False)

        self.other_Question = QPushButton('Other', self)
        self.other_Question.setFixedWidth(300)
        self.other_Question.setFixedHeight(50)
        self.other_Question.setStyleSheet("background-color: grey; font-size: 20px ")
        self.other_Question.setEnabled(False)

        self.delete_Question = QPushButton('Delete', self)
        self.delete_Question.setFixedWidth(300)
        self.delete_Question.setFixedHeight(50)
        self.delete_Question.setStyleSheet("background-color: red; font-size: 20px ")
        self.delete_Question.setEnabled(False)

        self.yesNoTotal = QLabel('...')
        self.yesNoTotal.setAlignment(Qt.AlignCenter)
        self.otherTotal = QLabel('...')
        self.otherTotal.setAlignment(Qt.AlignCenter)
        self.deletedTotal = QLabel('...')
        self.deletedTotal.setAlignment(Qt.AlignCenter)

        self.file_open.clicked.connect(self.openFile)
        self.yes_No_Question.clicked.connect(self.defineYesNo)
        self.other_Question.clicked.connect(self.defineOther)
        self.delete_Question.clicked.connect(self.defineDelete)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(title, 1, 0, 1, 6)
        grid.addWidget(self.file_open, 2, 5, 1, 1)
        grid.addWidget(self.file_name, 2, 0, 1, 4)
        grid.addWidget(question_number_label, 3, 0, 1, 1)
        grid.addWidget(self.question_number, 3, 1, 1, 1)
        grid.addWidget(remaining_questions_label, 3, 2, 1, 1)
        grid.addWidget(self.remaining_questions, 3, 3, 1, 1)
        grid.addWidget(total_questions_label, 3, 4, 1, 1)
        grid.addWidget(self.total_questions, 3, 5, 1, 1)
        grid.addWidget(self.question, 4, 0, 1, 6)
        grid.addWidget(self.yes_No_Question, 5, 0, 1, 2)
        grid.addWidget(self.other_Question, 5, 2, 1, 2)
        grid.addWidget(self.delete_Question, 5, 4, 1, 2)
        grid.addWidget(self.yesNoTotal, 6, 0, 1, 2)
        grid.addWidget(self.otherTotal, 6, 2, 1, 2)
        grid.addWidget(self.deletedTotal, 6, 4, 1, 2)

        widget = QWidget(self)
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        self.setGeometry(575, 300, 900, 200)
        self.setWindowTitle('Yes-No Question Classificator')
        self.statusBar().showMessage('Please open a file!')
        self.setWindowIcon(QIcon('question.png'))

        self.show()


    def openFile(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))

        fname = QFileDialog.getOpenFileName(self, 'Open file', current_directory, filter='csv(*.csv)')
        if fname[0]:
            self.statusBar().showMessage('Selected file: ' + fname[0])
            self.file_name.setText(fname[0])
            self.full_input_path = fname[0]
            self.file = fname[0].rsplit('/', 1)[1]
            self.directory = fname[0].rsplit('/', 1)[0]
            self.loadFile()
            self.create_ouput_dir()
            self.start_classification()

    def loadFile(self):
        self.raw_questions = pd.read_csv(self.file_name.text(), sep=';', encoding='latin-1', header=None)

    def create_ouput_dir(self):
        output_dir = self.directory + '/output_' + self.file.rsplit('.')[0]
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        os.chdir(output_dir)


    def defineYesNo(self):
        with open('classified.csv', 'a', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=';')
            try:
                writer.writerow([self.question.text(), 1])
                self.statusBar().showMessage('Question saved.')
            except:
                self.statusBar().showMessage('Skipped due to unexpected characters.')
        self.yesNoTotal.setText(str(int(self.yesNoTotal.text()) + 1))
        self.load_next()


    def defineOther(self):
        with open('classified.csv', 'a', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=';')
            try:
                writer.writerow([self.question.text(), 0])
                self.statusBar().showMessage('Question saved.')
            except:
                self.statusBar().showMessage('Skipped due to unexpected characters.')
        self.otherTotal.setText(str(int(self.otherTotal.text()) + 1))
        self.load_next()


    def defineDelete(self):
        with open('deleted.csv', 'a', newline='') as output_file:
            writer = csv.writer(output_file, delimiter=';')
            try:
                writer.writerow([self.question.text()])
                self.statusBar().showMessage('Question saved.')
            except:
                self.statusBar().showMessage('Skipped due to unexpected characters.')
        self.deletedTotal.setText(str(int(self.deletedTotal.text()) + 1))
        self.load_next()


    def load_next(self):
        try:
            self.question.setText((self.raw_questions[0][self.current_question]).strip())
            self.question_number.setText(str(self.current_question))
            self.remaining_questions.setText(str(self.total - self.current_question))
            self.current_question += 1
        except:
            self.statusBar().showMessage('End of file reached.')
            os.rename(self.full_input_path, self.full_input_path.rsplit('.', 1)[0] + 'completed.csv')



    def start_classification(self):
        self.yes_No_Question.setEnabled(True)
        self.other_Question.setEnabled(True)
        self.delete_Question.setEnabled(True)
        self.total = self.raw_questions[0].size
        self.current_question = 0
        self.total_questions.setText(str(self.total))
        self.yesNoTotal.setText('0')
        self.otherTotal.setText('0')
        self.deletedTotal.setText('0')

        self.load_next()



app = QApplication(sys.argv)
ex = Classification_interface()
sys.exit(app.exec_())