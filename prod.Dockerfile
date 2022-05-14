FROM python:3.10.4

#RUN apt-get update \
# && apt-get install -y sudo

#RUN adduser --disabled-password --gecos '' docker
#RUN adduser docker sudo
#
#USER docker

WORKDIR /code

ENV PYTHONUNBUFFERED 1

COPY ./requirements.txt /code/requirements.txt

RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY ./env/db.env /code/env/db.env

COPY ./tests /code/tests

#COPY ./api /code/api

CMD ["pytest"]