FROM python:3.11-bookworm

WORKDIR /usr/src

RUN pip install --no-cache-dir toml-to-requirements

COPY pyproject.toml .
RUN toml-to-req --toml-file pyproject.toml > requirements.txt
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir meson-python

#RUN apt-get update && \
#    apt-get install -y --no-install-recommends flatbuffers-compiler

COPY . .
RUN pip install .

ENTRYPOINT ["anmlriddle-mnist"]

