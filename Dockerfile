# we are building off the latest tf image with gpu
# and jupyter support baked in.
FROM tensorflow/tensorflow:latest-gpu-jupyter
RUN mkdir -p /home/dev/scratch/cars/carracing_clean
WORKDIR /home/dev/scratch/cars/carracing_clean
COPY ./flaskapp flaskapp
COPY ./agents agents
COPY ./keras_trainer keras_trainer
COPY ./requirements.txt requirements.txt
RUN apt update
RUN apt install -y git
RUN git clone https://github.com/openai/gym gym_latest
COPY ./cp_to_gym_envs/ gym_latest/gym/envs/
# This is to fix a bug in gym video recording. If it has been fixed, you can remove.
RUN sed -i '303s/\ \ \ \ \ \ \ \ /\ \ \ \ /' gym_latest/gym/wrappers/monitoring/video_recorder.py
RUN mkdir -p flaskapp/static
COPY ./cp_to_flaskapp_static flaskapp/static 
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get -y install tzdata 
RUN apt install swig xvfb ffmpeg x264  libcairo2-dev pkg-config python3-dev python3-pycurl libgirepository1.0-dev python3-tk python-opengl python3-pip -y
RUN pip3 install six
RUN pip3 install requests
RUN pip3 install -r requirements.txt
RUN pip3 install flask_pymongo pymongo gunicorn
RUN cd gym_latest && pip3 install -e .
RUN cd .. && mkdir -p train_logs
EXPOSE 5000
COPY gunicorn_start.sh .
RUN chmod +x gunicorn_start.sh
CMD ["/bin/bash", "gunicorn_start.sh"]

