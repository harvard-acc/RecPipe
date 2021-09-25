FROM nvidia/cuda:8.0-cudnn6-devel-ubuntu16.04

RUN apt-get update && apt-get install -y \
  python \
  python-pip \
  tmux

# Make pip happy about itself.
RUN pip install --upgrade pip

# Unlike apt-get, upgrading pip does not change which package gets installed,
# (since it checks pypi everytime regardless) so it's okay to cache pip.
RUN pip install matplotlib

RUN pip install future
RUN pip install numpy
RUN pip install onnx
RUN pip install pydot
RUN pip install torch
RUN pip install torchviz
RUN pip install scikit-learn
RUN pip install --upgrade sklearn
RUN pip install tqdm

RUN pip install torch

