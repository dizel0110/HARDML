FROM python:3.8.5

COPY . /app
WORKDIR /app

RUN pip install flask mlflow==1.14.1 boto3==1.20.19 scikit-learn==0.24.1

ENV FLASK_APP=app.py
EXPOSE 8000

CMD ["flask", "run", "-h", "0.0.0.0", "-p", "8000"]