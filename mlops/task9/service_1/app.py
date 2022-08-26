import os
import flask
import numpy
import models

from flask import request


app = flask.Flask(__name__)


@app.route('/get_source_iris_pred')
def get_source_iris_pred():
    sepal_length = request.args.get("sepal_length")
    sepal_width = request.args.get("sepal_width")
    o = numpy.array([[sepal_length, sepal_width]])

    prediction = models.model_float.predict(o)
    return flask.jsonify({'prediction': prediction[0]})


@app.route('/get_string_iris_pred')
def get_string_iris_pred():
    return "", 501
