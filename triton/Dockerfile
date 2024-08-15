FROM nvcr.io/nvidia/tritonserver:24.04-py3
RUN mkdir /triton
WORKDIR /triton/
ADD . /files/
RUN pip install -r requirements.txt