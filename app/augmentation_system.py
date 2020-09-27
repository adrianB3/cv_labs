import fnmatch
import inspect
import os
import cv2
import sys
import yaml

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt, QSize
from PyQt5.QtWidgets import QApplication, QLabel, QFileDialog, QMainWindow, QTextEdit, QAction, QToolBar, QStatusBar, \
    QProgressBar
from pathlib import Path

from augmentation.data_types import Data
from augmentation.pipeline import Pipeline

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

        cfg_file = open('augmentation_cfg.yml')
        self.cfg = yaml.load(cfg_file, Loader=yaml.FullLoader)

        self.input_img_dir = None
        self.output_img_dir = Path("D:\\dev\\cv_labs_test")  # TODO - select dir and apply _aug postfix
        self.filter_pattern = "*.jpg"
        self.img_paths = list()

        self.pipelines = []
        self.configurePipeline()

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
            counter = 1
            for img_path in self.img_paths:
                image = cv2.imread(img_path)
                image_name = os.path.basename(img_path).split(".")[0]
                new_path = os.path.join(self.output_img_dir, image_name + "_aug.jpg")
                self.text_out.append("Writing " + str(new_path))
                scaled_counter = scale(counter, [0, len(self.img_paths) * len(self.pipelines) - 1], [0, 100])
                self.pbar.setValue(int(scaled_counter))

                for pipeline in self.pipelines:
                    pipeline.execute(Data(image=image.copy(),
                                          file_path=img_path,
                                          output_dir=self.output_img_dir,
                                          count=counter,
                                          applied_augmentations=[]
                                          ))
                    counter += 1

                QApplication.processEvents()
        else:
            pass
        self.apply_aug_button.setChecked(False)

    def configurePipeline(self):
        self.text_out.append("CONFIGURATION:")
        self.text_out.append(yaml.dump(self.cfg))
        for transform_chain_name, transforms in self.cfg.items():
            pipeline = Pipeline(transform_chain_name)
            for transform in transforms:
                for transform_name, props in transform.items():
                    module = __import__('augmentation.transforms', fromlist=['object'])
                    for name, obj in inspect.getmembers(module):
                        if name == transform_name:
                            if props is not None:
                                pipeline.add_augmentation(obj(props))
                            else:
                                pipeline.add_augmentation(obj())
            self.pipelines.append(pipeline)


def main():
    app = QApplication(sys.argv)
    aug_sys = AugmentationSystem()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
