from PyQt5.QtCore import QObject, pyqtSignal


class EngineSignals(QObject):
    success = pyqtSignal(int, str)
    unconfirmed = pyqtSignal(list)
    error = pyqtSignal(str)
    progress = pyqtSignal(int, str)
