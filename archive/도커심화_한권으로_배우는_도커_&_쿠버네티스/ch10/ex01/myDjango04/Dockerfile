FROM python:3.11.6

WORKDIR /usr/src/app

COPY . .

RUN python -m pip install --upgrade pip	
RUN pip install -r requirements.txt

WORKDIR ./myapp	

CMD gunicorn --bind 0.0.0.0:8000 myapp.wsgi:application

EXPOSE 8000
