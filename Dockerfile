FROM pytorch/pytorch:latest
WORKDIR /deepedge/

COPY ./ ./
RUN pip install -r requirements.txt