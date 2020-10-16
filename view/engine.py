import asyncio
import time

from PyQt5.QtCore import QRunnable, pyqtSlot

from view.engine_callback import EngineSignals


class Engine(QRunnable):
    def __init__(self, img_path, label_path, *args, **kwargs):
        super(Engine, self).__init__()
        self.img_path = img_path
        self.label_path = label_path
        self.signals = EngineSignals()
        self.args = args
        self.kwargs = kwargs

    @pyqtSlot()
    def run(self):
        try:
            for i in range(0, 100):
                self.signals.progress.emit(i, 'in progress')
                time.sleep(1)
        except:
            self.signals.error.emit(100, 'SUCCESS')
        else:
            pass
        finally:
            self.signals.success.emit(100, 'SUCCESS')
