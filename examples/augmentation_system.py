import fnmatch
import os
import augmentation
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog, QMainWindow, QTextEdit, QAction, QToolBar, QStatusBar
from pathlib import Path


class AugmentationSystem(QMainWindow):

    def __init__(self):
        super().__init__()

        label = QLabel("CV-labs Augmentation System")
        label.setAlignment(Qt.AlignCenter)
        self.setCentralWidget(label)

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        open_folder_button = QAction(QIcon("assets/open_folder.png"), "Open Folder", self)
        open_folder_button.setStatusTip("Open image folder")
        open_folder_button.triggered.connect(self.show_dialog)
        open_folder_button.setCheckable(True)
        toolbar.addAction(open_folder_button)

        toolbar.addSeparator()

        self.setGeometry(300, 300, 550, 450)
        self.setWindowTitle('Augmentation System')
        self.setStatusBar(QStatusBar(self))
        self.show()

        self.input_img_dir = None
        self.output_img_dir = None
        self.filter_pattern = "*.jpg"
        self.img_paths = list()

    def show_dialog(self):
        home_dir = str(Path.home())
        dir_path = Path(QFileDialog.getExistingDirectory(self, 'Open folder', home_dir))
        if os.path.exists(dir_path):
            self.input_img_dir = dir_path
            self.get_valid_images()
            print("Found " + str(len(self.img_paths)) + " images.")
        else:
            raise FileNotFoundError(dir_path + "doesn't exist.")

    def get_valid_images(self):
        for entry in os.listdir(self.input_img_dir):
            if fnmatch.fnmatch(entry, self.filter_pattern):
                self.img_paths.append(os.path.join(self.input_img_dir, entry))


def main():
    app = QApplication(sys.argv)
    aug_sys = AugmentationSystem()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
