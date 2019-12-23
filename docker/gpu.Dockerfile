FROM pytorch/pytorch:1.3-cuda10.0-cudnn7-runtime

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

ADD ./dist/wsl_survey-0.1.0-py2.py3-none-any.whl /app/
WORKDIR /app
RUN pip install --quiet wsl_survey-0.1.0-py2.py3-none-any.whl --target .
ADD ./docker/entrypoint.sh /opt/
RUN pip install scipy torchnet tqdm torchvision


CMD ["/opt/entrypoint.sh"]
