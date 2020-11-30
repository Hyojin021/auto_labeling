import asyncio
import time

from PyQt5.QtCore import QRunnable, pyqtSlot

from engine.active_learning import ActiveLearning
from view.engine_callback import EngineSignals


class Engine(QRunnable):
    def __init__(self, img_path, label_path, *args, **kwargs):
        super(Engine, self).__init__()
        self.signals = EngineSignals()
        self.args = args
        self.kwargs = kwargs

    def set_path(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path

    @pyqtSlot()
    def run(self):

        if self.img_path is None or self.label_path is None:
            self.signals.error.emit(0, 'Path is None')
        try:
            # al = ActiveLearning()
            # al.run(self.img_path, self.label_path, self.signals)
            # self.signals.success.emit(100, 'SUCCESS')
            img_list = ['1.jpg', '2.jpg', '3.jpg', '4.jpg']
            self.signals.unconfirmed.emit(img_list)
        except:
            self.signals.error.emit(0, '')


