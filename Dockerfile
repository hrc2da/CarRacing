# we are building off the latest tf image with gpu
# and jupyter support baked in.
FROM tensorflow/tensorflow:latest-gpu-jupyter

COPY . /carracing
RUN apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install tzdata
RUN apt install swig xvfb libcairo2-dev pkg-config python3-dev python3-pycurl libgirepository1.0-dev python3-tk -y
RUN pip install six
RUN cd /carracing && pip install -r requirements.txt
RUN cd /carracing/gym_latest && pip install -e .
RUN mkdir -p train_logs
EXPOSE 5000
CMD ['/bin/bash']

# MKDR monitor-folder

