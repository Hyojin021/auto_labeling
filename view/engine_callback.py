from PyQt5.QtCore import QObject, pyqtSignal


class EngineSignals(QObject):
    success = pyqtSignal(int, str)
    error = pyqtSignal(int, str)
    progress = pyqtSignal(int, str)
