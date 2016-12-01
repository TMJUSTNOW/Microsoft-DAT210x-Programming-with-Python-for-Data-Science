import os
import sys

import PyQt5

dirname = os.path.dirname(PyQt5.__file__)
print(dirname)
plugin_path = os.path.join(dirname, 'Qt', 'plugins', 'platforms')
print(plugin_path)
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path