application_name = 'Self Driving Car Semantic Segmentation'
# pyqt packages
from PyQt5 import uic
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, QPointF
from PyQt5.QtWidgets import QMainWindow, QApplication, QMessageBox, QFileDialog, QLabel

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import sys
import numpy as np
import pickle
import torch
from PIL import Image

from unet import UNet
from main import BidirectionalMap, compute_device, inference, get_transform


def dark_JET_cmap():
    jet = plt.colormaps['jet']
    colors = jet(np.linspace(0.15, 0.9, 256))
    colors = np.vstack((np.array([0, 0, 0, 1]), colors))
    return LinearSegmentedColormap.from_list('modified_jet', colors)



def show_message(parent, title, message, icon=QMessageBox.Warning):
        msg_box = QMessageBox(icon=icon, text=message)
        msg_box.setWindowIcon(parent.windowIcon())
        msg_box.setWindowTitle(title)
        msg_box.setStyleSheet(parent.styleSheet() + 'color:white} QPushButton{min-width: 80px; min-height: 20px; color:white; \
                              background-color: rgb(91, 99, 120); border: 2px solid black; border-radius: 6px;}')
        msg_box.exec()
        


class QT_Action(QMainWindow):
    mouse_move_signal = pyqtSignal()
    
    def __init__(self):
        # system variable
        super(QT_Action, self).__init__()
        uic.loadUi('qt_main.ui', self)
        self.setWindowTitle(application_name) # set the title
        self.mouse_pos = None
        
        # runtime variable
        self.predicted = None
        self.image = None
        self.model = None
        self.transform = get_transform()['img']
        with open('class_ind_pair.pkl', 'rb') as f:
            self.class_ind_pair = pickle.load(f)
            
            
        # load the model
        self.load_model_action()
        
        
    def mouseMoveEvent(self,event):
        widget = self.childAt(event.pos())
        
        if widget is None:
            return 
        
        if isinstance(widget, QLabel) and widget.objectName == self.label_heatmap.objectName:
            mouse_pos = widget.mapFromGlobal(event.globalPos())
            x, y = mouse_pos.x(), mouse_pos.y()    
            self.mouse_pos = QPointF(x/widget.width(), y/widget.height())
            self.mouse_move_signal.emit()
            
            
    # linking all button/textbox with actions    
    def link_commands(self,):
        self.toolButton_import.clicked.connect(self.import_action)
        #self.comboBox_model.activated.connect(self.load_model_action)
        self.toolButton_process.clicked.connect(self.process_action)
        self.mouse_move_signal.connect(self.mouse_move_action)
        
        
    
    # choosing between models
    def load_model_action(self,):
        self.model_name = self.comboBox_model.currentText()
        
        # load the model
        if self.model_name == 'UNet':
            # load the model architechture
            self.model = UNet(3, len(self.class_ind_pair))
            
            # loading the training model weights
            self.model.load_state_dict(torch.load(f'{self.model_name}.pth'))
            
        # move model to GPU
        self.model = self.model.to(compute_device())
        
        self.model.eval() # Set model to evaluation mode
    
        
        
    
    # clicking the import button action
    def import_action(self,):
        # show an "Open" dialog box and return the path to the selected file
        filename, _ = QFileDialog.getOpenFileName(None, "Select file", options=QFileDialog.Options())
        self.lineEdit_import.setText(filename)
        
        # didn't select any files
        if filename is None or filename == '': 
            return
    
        # selected .png files
        if filename.endswith('.png'):
            self.image = Image.open(filename) 
            self.lineEdit_import.setText(filename)
            #X = [transform(img)]
            self.update_display()
        
        # selected the wrong file format
        else:
            show_message(self, title='Load Error', message='Available file format: .png')
            self.import_action()
        
        
    def update_display(self):
        if not self.image is None:
            image = np.array(self.image).astype(np.uint8)
            h, w, ch = image.shape
            q_image = QImage(image.data.tobytes(), w, h, ch*w, QImage.Format_RGB888)  # Create QImage
            qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
            self.label_image.setPixmap(qpixmap)
            
            
    def process_action(self):
        if self.image is None:
            show_message(self, title='Process Error', message='Please load an image first')
            return
        
        # apply the transform
        data = self.transform(self.image)
        
        # inference
        predicted = inference(self.model, data)
        
        self.predicted = predicted.squeeze().numpy()
        heatmap = self.predicted / len(self.class_ind_pair)
        heatmap_colored = dark_JET_cmap()(heatmap) * 255
        
        # Create QImage from the heatmap
        q_image = QImage(heatmap_colored.astype(np.uint8), heatmap_colored.shape[1], heatmap_colored.shape[0], QImage.Format_RGBA8888)
        
        qpixmap = QPixmap.fromImage(q_image)  # Convert QImage to QPixmap
        self.label_heatmap.setPixmap(qpixmap)
        
    
    def mouse_move_action(self):
        if self.predicted is None:
            return
        posX, posY = self.mouse_pos.x(), self.mouse_pos.y()
        posX, posY = int(posX*self.predicted.shape[1]), int(posY*self.predicted.shape[0])
        text = self.class_ind_pair.get_value(self.predicted[posY, posX])
        self.lineEdit_prediction.setText(text)
        
        
def main():
    app = QApplication(sys.argv)
    action = QT_Action()
    action.link_commands()
    action.show()
    sys.exit(app.exec_())
    
    
if __name__ == '__main__':
    main()