from PyQt5.QtWidgets import (QMainWindow, QWidget, QPushButton, QFileDialog, qApp, QApplication, QLabel, QLineEdit, QGridLayout, QAction)
from PyQt5.QtGui import QIntValidator, QFont, QIcon, QPalette, QColor
from PyQt5.QtCore import Qt
import pandas as pd
import sys
import os



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

        self.loadFile()

        title_font = QFont('Arial', 20)
        title_font.setBold(True)
        edit_font = QFont('Arial', 20)

        title = QLabel('Yes-No Classifier Interface')
        title.setAlignment(Qt.AlignCenter)
        title.setFont(title_font)


        self.file_label = QLabel('Opened file: ')
        self.file_label.setStyleSheet("font-size: 14px")

        self.file_name = QLabel('...')
        self.file_name.setStyleSheet("font-size: 14px")



        self.question = QLineEdit()
        self.question.setFixedHeight(50)
        self.question.setFont(edit_font)

        self.yes_No_Question = QPushButton('Yes-No', self)
        self.yes_No_Question.setFixedWidth(300)
        self.yes_No_Question.setFixedHeight(50)
        self.yes_No_Question.setStyleSheet("background-color: green; font-size: 20px ")

        self.other_Question = QPushButton('Other', self)
        self.other_Question.setFixedWidth(300)
        self.other_Question.setFixedHeight(50)
        self.other_Question.setStyleSheet("background-color: grey; font-size: 20px ")


        self.delete_Question = QPushButton('Delete', self)
        self.delete_Question.setFixedWidth(300)
        self.delete_Question.setFixedHeight(50)
        self.delete_Question.setStyleSheet("background-color: red; font-size: 20px ")



        self.yes_No_Question.clicked.connect(self.defineYesNo)
        self.other_Question.clicked.connect(self.defineOther)
        self.delete_Question.clicked.connect(self.defineDelete)

        grid = QGridLayout()
        grid.setSpacing(10)
        grid.addWidget(title, 1, 0, 1, 6)
        grid.addWidget(self.file_label, 2, 0, 1, 2)
        grid.addWidget(self.file_name, 2, 1, 1, 2)
        grid.addWidget(self.question, 3, 0, 1, 6)
        grid.addWidget(self.yes_No_Question, 4, 0, 1, 2)
        grid.addWidget(self.other_Question, 4, 2, 1, 2)
        grid.addWidget(self.delete_Question, 4, 4, 1, 2)

        widget = QWidget(self)
        widget.setLayout(grid)
        self.setCentralWidget(widget)

        self.setGeometry(575, 300, 900, 200)
        self.setWindowTitle('Yes-No Question Classificator')
        self.statusBar().showMessage('Please open a file!')
        self.show()


    def openFile(self):
        current_directory = os.path.dirname(os.path.abspath(__file__))

        fname = QFileDialog.getOpenFileName(self, 'Open file', current_directory, filter='csv(*.csv)')
        if fname[0]:
            self.statusBar().showMessage('Selected file: ' + fname[0])
            self.file_name.setText(fname[0])


    def loadFile(self):
        self.raw_questions = pd.read_csv("tweet.csv", sep='\t')
        pass

    def defineYesNo(self):
        pass

    def defineOther(self):
        pass

    def defineDelete(self):
        pass



app = QApplication(sys.argv)
ex = Classification_interface()
sys.exit(app.exec_())