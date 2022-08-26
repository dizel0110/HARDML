import os
import flask
import signal

import external
import services


APP_MODE: str = os.environ['APP_MODE']
APP_NAME: str = services.gen_replica_name()
APP_HOST: str = os.environ['APP_HOST']
APP_PORT: str = os.environ['APP_PORT']

EXTERNAL_URL: str = os.environ['EXTERNAL_URL']
EXTERNAL_MAX_RETRIES: int = int(os.environ['EXTERNAL_MAX_RETRIES'])

SD_REDIS_HOST: str = os.environ['SD_REDIS_HOST']
SD_REDIS_PORT: int = int(os.environ['SD_REDIS_PORT'])
SD_REDIS_PASSWORD: str = os.environ['SD_REDIS_PASSWORD']
SD_REPLICAS_KEY: str = os.environ['SD_REPLICAS_KEY']


sd = services.ServiceDiscovery(SD_REDIS_HOST,
                               SD_REDIS_PORT,
                               SD_REDIS_PASSWORD,
                               SD_REPLICAS_KEY, )


signal.signal(signal.SIGINT, lambda sig, fra: sd.unregister(APP_NAME))
signal.signal(signal.SIGTERM, lambda sig, fra: sd.unregister(APP_NAME))


app = flask.Flask(__name__)


@app.route('/return_secret_number')
def get_secret_number():
   return flask.jsonify({'secret_number': external.secret_number})


if __name__ == '__main__':
   external.init_external(EXTERNAL_URL, EXTERNAL_MAX_RETRIES)

   sd.register(name=APP_NAME,
               parameters={'host': APP_HOST,
                           'port': APP_PORT, }, )
   
   app.run(debug=True if APP_MODE == 'debug' else False,
           host='0.0.0.0',
           port=8080)
