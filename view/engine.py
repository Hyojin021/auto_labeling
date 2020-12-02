from PyQt5.QtCore import QRunnable, pyqtSlot

from engine.active_learning import ActiveLearning
from view.engine_callback import EngineSignals


class Engine(QRunnable):
    def __init__(self, *args, **kwargs):
        super(Engine, self).__init__()
        self.signals = EngineSignals()
        self.args = args
        self.kwargs = kwargs
        self.setAutoDelete(False)
        self.al = ActiveLearning()

    def set_path(self, img_path, label_path):
        self.img_path = img_path
        self.label_path = label_path

    def set_active_learning(self, set_active_learning):
        self.set_active_learning = set_active_learning

    @pyqtSlot()
    def run(self):

        if self.img_path is None or self.label_path is None:
            self.signals.error.emit('Path is None')
        if self.set_active_learning:
            # TODO: 액티브 러닝 실행, 미완료 된 파일 있으면 self.signals.unconfirmed.emit(img_list) 호출,
            # TODO: 주의: img_list는 풀패스로 줘야함, 파일 이름만 주면 안됨
            try:
                # self.al.run(self.img_path, self.label_path, self.signals)
                # self.signals.success.emit(100, 'SUCCESS')
                # img_list = [
                #     '%s/1.jpg' % self.img_path,
                #     '%s/2.jpg' % self.img_path,
                #     '%s/3.jpg' % self.img_path,
                #     '%s/4.jpg' % self.img_path
                # ]
                # self.signals.unconfirmed.emit(img_list)
                raise Exception('ERRRORORRO')
            except Exception as e:
                self.signals.error.emit(str(e))
        else:
            # TODO: 추론
            pass
