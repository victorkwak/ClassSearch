FROM ubuntu:latest
MAINTAINER Luis Magana "maganaluis92@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN sh -c '/bin/echo -e "yes\n" | apt-get install python-tk'
RUN sh -c '/bin/echo -e "yes\n" | apt-get install wget'
COPY . /app
WORKDIR /app/fastText
RUN pip install pybind11
RUN apt-get install python-sklearn -y  
RUN python setup.py install
WORKDIR /app/models
RUN wget https://www.dropbox.com/s/5k5vket88csvn5i/fasttext.bin
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python"]
CMD ["app.py"]
