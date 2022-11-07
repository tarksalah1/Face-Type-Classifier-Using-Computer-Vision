from PyQt5 import QtWidgets as qtw
from PyQt5 import uic
import sys
import qdarkstyle
from PyQt5.QtCore import pyqtSlot
from FaceClassifier import FaceClassifier
# press "apply" for your face type

class MainWindow(qtw.QMainWindow):
    def __init__(self):
        super().__init__()
        uic.loadUi("src/ui/mainWindow.ui", self)
        self.setStyleSheet(qdarkstyle.load_stylesheet(qt_api='pyqt5'))
        self.setStyleSheet(qdarkstyle.load_stylesheet_pyqt5())
        self.faceClassifier = FaceClassifier()
        self.centralWidget().layout().addWidget(self.faceClassifier)
        self.browse_action.triggered.connect(self.loadFace)
        
    @pyqtSlot()
    def loadFace(self):
        image_path = qtw.QFileDialog.getOpenFileName(filter="Image (*.*)")[0]
        self.faceClassifier.loadFace(image_path)


if __name__ == '__main__':
    app = qtw.QApplication(sys.argv)
    main = MainWindow()
    main.show()
    sys.exit(app.exec_())
