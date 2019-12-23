FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY ./wsl_survey /app/
WORKDIR /app
ADD ./docker/entrypoint.sh /opt/
RUN pip install scipy torchnet tqdm torchvision

CMD ["/opt/entrypoint.sh"]
