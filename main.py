import utils

from flask import Flask, jsonify, request

app = Flask(__name__)

# Przykładowa początkowa baza danych
database = []

class_value = "99"

# Endpoint do pobierania wszystkich danych
@app.route('/data', methods=['GET'])
def get_data():
    return jsonify(data=database,
                   class_val=class_value)

# Endpoint do pobierania pojedynczego rekordu
@app.route('/data/<int:row_id>', methods=['GET'])
def get_single_data(row_id):
    if 0 <= row_id < len(database):
        data_with_class = {"data": database[row_id]["data"], "class": database[row_id]["class"]}
        return jsonify(data_with_class)
    return jsonify({'error': 'Row not found'})

# Endpoint do aktualizowania istniejącego rekordu
@app.route('/data/<int:row_id>', methods=['PUT'])
def update_data(row_id):
    if 0 <= row_id < len(database):
        updated_data = request.get_json()
        database[row_id] = updated_data
        return jsonify({'message': 'Row updated successfully'})
    return jsonify({'error': 'Row not found'})

# Endpoint do zamiany obecnej bazy na nową
@app.route('/data', methods=['POST'])
def replace_data():
    new_data = request.get_json()
    if isinstance(new_data, dict) and "data" in new_data and "class" in new_data:
        global database
        database = new_data["data"]
        global class_value
        class_value = new_data["class"]
        utils.nn_classification(database, class_value)
        return jsonify({'message': 'Database replaced successfully'})
    return jsonify({'error': 'Invalid data format'})

if __name__ == '__main__':
    app.run()