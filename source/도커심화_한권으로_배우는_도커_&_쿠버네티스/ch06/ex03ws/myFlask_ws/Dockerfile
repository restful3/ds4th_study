FROM python:3.11.6

WORKDIR /usr/src/app

COPY . .

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt

WORKDIR ./myapp

EXPOSE 8001

CMD ["gunicorn", "--bind", "0.0.0.0:8081", "main:app"]