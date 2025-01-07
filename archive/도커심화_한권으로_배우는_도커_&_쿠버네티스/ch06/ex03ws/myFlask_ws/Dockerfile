FROM python:3.11.6

WORKDIR /usr/src/app
RUN python -m pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 8081

CMD ["python", "main.py"]