# we are building off the latest tf image with gpu
# and jupyter support baked in.
FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir -p /home/dev/scratch/cars/carracing_clean
WORKDIR /home/dev/scratch/cars/carracing_clean
COPY ./flaskapp flaskapp
COPY ./agents agents
COPY ./keras_trainer keras_trainer
COPY ./requirements.txt requirements.txt
RUN git clone https://github.com/openai/gym gym_latest
COPY ./cp_to_gym_envs/ gym_latest/gym/envs/
RUN mkdir -p flaskapp/static
COPY ./cp_to_flaskapp_static flaskapp/static 
RUN apt update
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install tzdata 
RUN apt install swig xvfb libcairo2-dev pkg-config python3-dev python3-pycurl libgirepository1.0-dev python3-tk ffmpeg python-opengl -y
RUN pip3 install six
RUN pip3 install requests
RUN pip3 install -r requirements.txt
RUN pip3 install flask_pymongo pymongo keras tensorflow
RUN cd gym_latest && pip3 install -e .
RUN cd .. && mkdir -p train_logs
EXPOSE 5000
RUN ls gym_latest/gym/envs/box2d
# CMD ["xvfb-run","-a","-s \"screen 1400x900x24\"","python3", "flaskapp/app.py"]
CMD ["/bin/bash"]
# MKDR monitor-folder

