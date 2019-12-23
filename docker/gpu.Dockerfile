FROM pytorch/pytorch:1.3-cuda10.1-cudnn7-runtime


ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8
RUN apt-get update -y
RUN apt-get install -y libsm6 libxext6 libxrender-dev libglib2.0-0 \
        libgtk2.0-dev pkg-config libavcodec-dev libavformat-dev \
        libswscale-dev python-dev python-numpy libtbb2 libtbb-dev \
        libjpeg-dev libpng-dev libtiff-dev libjasper-dev libdc1394-22-dev
RUN pip install cython
RUN pip install scipy torchnet==0.0.4 torchvision==0.4.2 tqdm==4.41.0 \
    xmltodict==0.12.0 matplotlib==3.1.2 tensorboard==2.1.0 terminaltables==3.1.0 \
    colorama==0.4.3 opencv-python==4.1.2.30 scikit-learn==0.22.0  pydensecrf imageio

ADD ./dist/wsl_survey-0.1.0-py2.py3-none-any.whl /app/
ADD ./datasets/test /test_dataset
WORKDIR /app
RUN pip install --quiet wsl_survey-0.1.0-py2.py3-none-any.whl --target /app
ADD ./docker/entrypoint.sh /opt/

CMD ["/opt/entrypoint.sh"]
