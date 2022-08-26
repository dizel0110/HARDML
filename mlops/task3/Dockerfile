FROM --platform=amd64 python:3.6

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

CMD ["/usr/local/bin/uwsgi", \
     "--http", "0.0.0.0:9090", \
     "--wsgi-file", "app.py", \
     "--master", \
     "--processes", "2", \
     "--threads", "5", \
     "--lazy-apps", \
     "--master-fifo=/var/master.fifo"]