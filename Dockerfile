FROM ubuntu:latest
LABEL authors="omrilapidot"

FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python" , "vbet_agg_data_creator.py"]