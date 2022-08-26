FROM --platform=amd64 python:3.6

COPY . /app
WORKDIR /app

RUN pip install -r requirements.txt

ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0
EXPOSE 5000

CMD ["flask", "run"]