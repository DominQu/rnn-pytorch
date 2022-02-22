FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04
RUN apt update
RUN apt install python3 python3-pip
RUN python3 -m pip install torch==1.10.2

COPY . ~/rnn
CMD ["python3", "~/rnn/main.py"]

