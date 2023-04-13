FROM python:3.7.15

ADD . /workspace/spdet

WORKDIR /workspace/spdet
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
RUN apt-get update
RUN apt-get install libgl1
#
RUN sed -i "s/archive.ubuntu./mirrors.aliyun./g" /etc/apt/sources.list
RUN sed -i "s/deb.debian.org/mirrors.aliyun.com/g" /etc/apt/sources.list
RUN sed -i "s/security.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list
RUN sed -i "s/httpredir.debian.org/mirrors.aliyun.com\/debian-security/g" /etc/apt/sources.list
RUN pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip config set install.trusted-host pypi.tuna.tsinghua.edu.cn
#
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numpy
RUN pip install torch torchvision  torchaudio==0.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cython
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pycocotools
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple cython_bbox
#
RUN python setup.py develop