FROM python:3.10

RUN mkdir /flask_diplom

WORKDIR /flask_diplom

COPY requeriments.txt .

RUN pip install -r requeriments.txt

COPY . .


CMD [ "python", "main_flask.py" ]