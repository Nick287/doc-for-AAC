FROM ubuntu:18.04 as base

EXPOSE 8080

RUN apt-get update && export DEBIAN_FRONTEND=noninteractive \
    && apt-get -y install --no-install-recommends python3-pip python3-dev libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN cd /usr/local/bin
RUN ln -s /usr/bin/python3 python
# RUN apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev
RUN pip3 install --upgrade pip
RUN pip3 install numpy flask gunicorn
RUN pip3 install numpy pillow opencv-python
RUN pip3 install onnxruntime
RUN pip install azure-iot-device~=2.7.0
RUN pip3 install numpy scipy matplotlib
RUN apt-get install -y libgl1-mesa-dev && apt-get clean

# Copy the app file and the tags file
RUN mkdir /app
# COPY ./app/*.py /app/
COPY /*.py /app/

# Start http extension module 
FROM base as final
WORKDIR /app
ENTRYPOINT ["python3", "main.py"]