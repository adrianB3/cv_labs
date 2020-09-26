import fnmatch
import os

import cv2

import augmentation
import sys

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog, QMainWindow, QTextEdit, QAction, QToolBar, QStatusBar, \
    QProgressBar
from pathlib import Path

from augmentation.add_name_transform import AddMyName
from augmentation.data_types import Data
from augmentation.pipeline import Pipeline
from augmentation.write_image import WriteImage
from app.utils import scale


class AugmentationSystem(QMainWindow):

    def __init__(self):
        super().__init__()

        toolbar = QToolBar("Main Toolbar")
        toolbar.setIconSize(QSize(24, 24))
        self.addToolBar(toolbar)

        self.open_folder_button = QAction(QIcon("assets/open_folder.png"), "Opsen Folder", self)
        self.open_folder_button.setStatusTip("Open image folder")
        self.open_folder_button.triggered.connect(self.show_dialog)
        self.open_folder_button.setCheckable(True)
        toolbar.addAction(self.open_folder_button)

        toolbar.addSeparator()

        self.apply_aug_button = QAction(QIcon("assets/apply_aug.png"), "Apply", self)
        self.apply_aug_button.setStatusTip("Apply defined augmentations")
        self.apply_aug_button.triggered.connect(self.apply_augmentations)
        self.apply_aug_button.setCheckable(True)
        toolbar.addAction(self.apply_aug_button)

        toolbar.addSeparator()

        self.pbar = QProgressBar(self)
        self.pbar.setGeometry(50, 0, 100, 25)
        toolbar.addWidget(self.pbar)

        toolbar.addSeparator()

        self.setGeometry(300, 300, 550, 450)
        self.setWindowTitle('Augmentation System')
        self.setStatusBar(QStatusBar(self))

        self.text_out = QTextEdit(self)
        self.text_out.setReadOnly(True)
        self.text_out.append("Hello to Augmentation System!")
        self.text_out.setAlignment(Qt.AlignLeft)
        self.setCentralWidget(self.text_out)

        self.show()

        self.input_img_dir = None
        self.output_img_dir = Path("D:\\dev\\cv_labs_test")
        self.filter_pattern = "*.jpg"
        self.img_paths = list()

        self.pipeline = Pipeline()
        self.pipeline\
            .add_augmentation(AddMyName())\
            .add_augmentation(WriteImage())

    def show_dialog(self):
        home_dir = str(Path.home())
        dir_path = Path(QFileDialog.getExistingDirectory(self, 'Open folder', home_dir))
        if os.path.exists(dir_path):
            self.input_img_dir = dir_path
            self.get_valid_images()
            self.text_out.append("Found " + str(len(self.img_paths)) + " images.")
        else:
            raise FileNotFoundError(dir_path + "doesn't exist.")
        self.open_folder_button.setChecked(False)

    def get_valid_images(self):
        for entry in os.listdir(self.input_img_dir):
            if fnmatch.fnmatch(entry, self.filter_pattern):
                self.img_paths.append(os.path.join(self.input_img_dir, entry))

    def apply_augmentations(self):
        if len(self.img_paths) != 0:
            counter = 0
            for img_path in self.img_paths:
                image = cv2.imread(img_path)
                image_name = os.path.basename(img_path).split(".")[0]
                new_path = os.path.join(self.output_img_dir, image_name + "_aug.jpg")
                self.text_out.append("Writing " + str(new_path))
                scaled_counter = scale(counter, [0, len(self.img_paths) - 1], [0, 100])
                self.pbar.setValue(int(scaled_counter))

                self.pipeline.execute(Data(image=image, file_path=img_path, output_dir=self.output_img_dir, count=counter, applied_augmentations=[]))

                counter += 1
                QApplication.processEvents()
        else:
            pass
        self.apply_aug_button.setChecked(False)


def main():
    app = QApplication(sys.argv)
    aug_sys = AugmentationSystem()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
