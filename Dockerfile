FROM pytorch/pytorch:latest
WORKDIR /deepedge/

COPY ./ ./
RUN pip install --no-cache-dir torch torchvision cython matplotlib sklearn