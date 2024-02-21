FROM python:3.11

WORKDIR /usr/src

# Copy the script and any other necessary files into the container
COPY requirements.txt /usr/src

# Install any additional dependencies
RUN pip install -r requirements.txt

COPY . /usr/src

ENTRYPOINT ["python", "train.py"]

