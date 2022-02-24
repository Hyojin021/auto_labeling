from view.engine import Engine
import os

engine = Engine()

dir_path = os.getcwd()
img_path = dir_path + '/Image'
label_path = dir_path + '/Label/'
engine.set_path(img_path, label_path)
engine.set_active_learning(False)
engine.run()
