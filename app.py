from flask import Flask, request, send_from_directory
from flask import request
import os
#from .attnGAN.gen_art import gen_example_from_text
from user import about

project_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(project_dir, 'images')

app = Flask(__name__, static_url_path='')

@app.route("/")
def hello():
    return "Hello World!"

@app.route("/test")
def my_route():
    phrase = request.args.get('input', default = 'a red bird', type = str)
    #image_name = ml_model(phrase, output_dir, )
    return phrase

@app.route('/about')
def hey():
    return about(my_route())

if __name__ == '__main__':
    app.run(debug=True)
