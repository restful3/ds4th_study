from flask import Flask, render_template, send_from_directory
import os

app = Flask(__name__, template_folder='pages', static_folder='pages')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory(app.static_folder, path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8081)