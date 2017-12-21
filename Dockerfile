FROM ubuntu:latest
MAINTAINER Luis Magana "maganaluis92@gmail.com"
RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential
RUN sh -c '/bin/echo -e "yes\n" | apt-get install python-tk'
RUN sh -c '/bin/echo -e "yes\n" | apt-get install wget'
COPY . /app
RUN apt-get install git -y
WORKDIR /app
RUN git clone https://github.com/facebookresearch/fastText.git
WORKDIR /app/fastText
RUN pip install pybind11
RUN apt-get install python-sklearn -y  
RUN python setup.py install
WORKDIR /app/data
RUN tar -xzf cs_subs.csv.tar.gz
WORKDIR /app
RUN pip install -r requirements.txt
RUN mkdir models
RUN python build_model.py
ENTRYPOINT ["python"]
CMD ["app.py"]
