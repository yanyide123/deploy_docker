# 使用NVIDIA官方的CUDA 11.2基础镜像，该镜像已经包括了cuDNN
FROM nvidia/cuda:11.2.2-cudnn8-devel-ubuntu20.04

# 设置代理服务器,，没有代理可以不写
ENV http_proxy http://10.10.10.10:8080/
ENV https_proxy http://10.10.10.10:8080/

# 避免安装过程中的任何交互式询问
ARG DEBIAN_FRONTEND=noninteractive

#APT包管理器去从Ubuntu的密钥服务器上获取指定密钥的公钥，以便在安装软件包时进行验证。这里是安装nvidia的包
# RUN apt-key adv --keyserver keyserver.ubuntu.com --recv-keys A4B469963BF863CC
COPY nvidia-key.asc /tmp/
RUN apt-key add /tmp/nvidia-key.asc

# 更新软件包列表，安装python3及pip，并清理缓存
RUN apt-get update && \
    apt-get install -y python3-pip python3-dev && \
    rm -rf /var/lib/apt/lists/*

# 安装其他依赖
RUN apt-get update && apt-get install ffmpeg libsm6 libxext6  -y

# 使用update-alternatives设置默认的python和pip版本
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# 设置工作目录
WORKDIR /swin_tranformer
RUN mkdir /home/bml/app/swin_tranformer
# 将当前目录（即您的项目目录）内容复制到容器的/swin_tranformer中
COPY . /home/bml/app/swin_tranformer


# 使用阿里云镜像源安装特定版本的PyTorch（CPU版本）
RUN pip install networkx==3.1 -i https://mirrors.aliyun.com/pypi/simple/
RUN pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html

# 离线安装下载的包
#RUN python -m pip install ./packages/torch-2.2.2+cpu-cp38-cp38-linux_x86_64.whl
# 安装GDAL需要的插件
#RUN apt-get install build-essential libpq-dev gdal-bin libgdal-dev -y

# 使用阿里云镜像源安装requirements.txt中指定的所需包
RUN python3 -m pip install --no-cache-dir -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
RUN python3 -m pip install cryptography -i http://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.aliyun.com

# 切换到swin_tranformer目录并安装，下面是编译
#WORKDIR /swin_tranformer/mmaction
#RUN python setup.py install

# 创建用户
RUN groupadd -g 601 bml && useradd -m -s /bin/bash -N -u 601 bml -g bml
RUN mkdir /home/bml/app
RUN chown -R bml:bml /home/bml

# 给用户权限
RUN chown -R bml:bml /home/bml
USER bml
WORKDIR /home/bml

# 将端口8080暴露给容器外部的世界
EXPOSE 8080

# 定义环境变量
#ENV FLASK_APP=py_model_server.py
#ENV FLASK_RUN_HOST=0.0.0.0
#ENV FLASK_RUN_PORT=8081
#
## 使用flask命令运行app.py
#CMD ["flask", "run"]



