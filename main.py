import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel
from PyQt5.QtGui import QPixmap, QImage
from PyQt5 import uic
from ROC import ROCCurve  
import matplotlib.pyplot as plt
import io

class ROCApp(QMainWindow):
    def __init__(self):
        super(ROCApp, self).__init__()
        uic.loadUi(r"C:\Users\bassa\Downloads\TASK5\Task5.ui", self)  
        self.ROC.clicked.connect(self.load_data_and_plot_roc)

    def load_data_and_plot_roc(self):
        # Example data for now
        np.random.seed(42)
        y_true = np.random.randint(0, 2, 100)
        y_scores = y_true * 0.8 + np.random.normal(0, 0.5, 100)

        roc = ROCCurve()
        roc.compute_roc(y_true, y_scores)

        # Plot ROC on matplotlib
        fig, ax = plt.subplots(figsize=(5, 4))
        roc.plot_roc_curve(ax=ax, title="ROC Curve")

        # Save to a buffer
        buf = io.BytesIO()
        fig.savefig(buf, format='png')
        buf.seek(0)

        # Load image from buffer
        image = QImage()
        image.loadFromData(buf.read(), 'PNG')
        pixmap = QPixmap.fromImage(image)

        # Set image to QLabel
        self.OutputImage.setPixmap(pixmap)
        self.OutputImage.setScaledContents(True)

        plt.close(fig)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = ROCApp()
    window.show()
    sys.exit(app.exec_())
