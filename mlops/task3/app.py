import time
import flask


app = flask.Flask(__name__)


@app.route('/return_secret_number')
def hello():
    time.sleep(1)
    return flask.jsonify({'secret_number': 0})


if __name__ == '__main__':
    app.run()
else:
    application = app
