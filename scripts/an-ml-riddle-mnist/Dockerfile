FROM python:3.11-bookworm

WORKDIR /usr/src

RUN pip install --no-cache-dir hatch

COPY pyproject.toml .
RUN hatch dep show requirements > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN apt-get update && \
    apt-get install -y --no-install-recommends flatbuffers-compiler

COPY . .
RUN pip install .

ENTRYPOINT ["anmlriddle-mnist"]

