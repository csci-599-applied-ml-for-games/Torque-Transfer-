FROM python:3.7-buster
WORKDIR /code
ENV FLASK_APP app.py
ENV FLASK_RUN_HOST 0.0.0.0
ENV FLASK_RUN_PORT 5000
RUN apt-get update
RUN apt-get install -y gcc musl-dev git libglib2.0-dev libgl1-mesa-dev libglu1-mesa-dev freeglut3-dev libzmq5-dev libplib-dev libopenal-dev libalut-dev libxi-dev libxmu-dev libxrender-dev libxrandr-dev libpng-dev libvorbis-dev xvfb ffmpeg libopencv-dev
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY . .
RUN sh setup.sh