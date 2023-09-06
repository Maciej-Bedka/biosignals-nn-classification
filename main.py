import utils
import ast
from PyQt5 import QtWidgets
from PyQt5 import QtGui
from PyQt5 import QtCore
from pyqtgraph import PlotWidget, plot
import pyqtgraph as pg
import sys
import threading
import ctypes


from flask import Flask, jsonify, request

appRest = Flask(__name__)


# Przykładowa początkowa baza danych
database = []
predicted_values = []
classes = [0,1,2,3,4,5,6]
class_value = "99"


# Endpoint do zamiany obecnej bazy na nową
@appRest.route('/data', methods=['POST'])
def replace_data():
    new_data = request.get_json()
    if isinstance(new_data, dict) and "data" in new_data and "class" in new_data:
        global database
        database = new_data["data"]
        global class_value
        class_value = new_data["class"]
        global predicted_values 
        predicted_values = utils.nn_classification(database, class_value)
        predicted_values = predicted_values.flatten()
        w.update_graph(predicted_values)
        
        return jsonify({'message': 'Database replaced successfully'})
    return jsonify({'error': 'Invalid data format'})


def get_id(thread_app):
    if hasattr(thread_app, '_thread_id'):
        return thread_app._thread_id
    for id, thread in threading._active.items():
        if thread is thread_app:
            return id


def raise_exception(thread_app):
    thread_id = get_id(thread_app)
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id,
          ctypes.py_object(SystemExit))
    if res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(thread_id, 0)
        print('Exception raise failure')

def start_service():
    appRest.run(port=5000)

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, predicted_values, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        

        self.graphWidget = pg.PlotWidget()

        self.setCentralWidget(self.graphWidget)

        classes = [0,1,2,3,4,5,6]

        self.graphWidget.plot(classes, predicted_values)

    def update_graph(self, predicted_values):
        bargraph = pg.BarGraphItem(x = classes, height = predicted_values, width = 0.6, brush ='g')
        self.graphWidget.clear()
        self.graphWidget.addItem(bargraph)
        # self.graphWidget.plot(classes, predicted_values)
        self.show()

    def UiComponents(self):
        button = QtWidgets.QPushButton("CLICK", self)


        button.clicked.connect(self.clickme)

    def clickme(self):
        print("pressed")
        

if __name__ == '__main__':
    rest_thread = threading.Thread(target=start_service)
    rest_thread.start()
    app = QtWidgets.QApplication(sys.argv)
    w = MainWindow(predicted_values=predicted_values)
    # communication.rest.on_data_received = w.on_data_received
    w.show()
    app.exec_()
    raise_exception(rest_thread)
    # show_gui()