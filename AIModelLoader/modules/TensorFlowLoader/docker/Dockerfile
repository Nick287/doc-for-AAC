FROM tensorflow/tensorflow:2.7.0 as base

EXPOSE 8080

RUN apt-get update && apt-get install -y --no-install-recommends 
RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev

RUN pip install --upgrade pip
RUN pip install numpy flask gunicorn
RUN pip install numpy pillow opencv-python
RUN pip install requests protobuf grpcio
RUN pip install numpy scipy matplotlib
RUN pip install azure-iot-device~=2.7.0
RUN apt update
RUN apt-get install -y libgl1-mesa-dev && apt-get clean

# Copy the app file and the tags file
RUN mkdir /app
# COPY ./app/*.py /app/
COPY /*.py /app/

# Start http extension module 
FROM base as final
WORKDIR /app
ENTRYPOINT ["python3", "main.py"]
