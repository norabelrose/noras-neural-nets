# 1) choose base container
# generally use the most recent tag

# data science notebook
# https://hub.docker.com/repository/docker/ucsdets/datascience-notebook/tags
ARG BASE_CONTAINER=ucsdets/datascience-notebook:2020.2-stable

# scipy/machine learning (tensorflow)
# https://hub.docker.com/repository/docker/ucsdets/scipy-ml-notebook/tags
# ARG BASE_CONTAINER=ucsdets/scipy-ml-notebook:2020.2-stable

FROM $BASE_CONTAINER

LABEL maintainer="UC San Diego ITS/ETS <ets-consult@ucsd.edu>"

# 2) change to root to install packages
USER root

RUN	apt-get install htop

RUN apt-get update && \
	apt-get install -y \
			libtinfo5

# Unlike the scipy-ml container, we use CUDA 11
wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda_11.1.1_455.32.00_linux.run
sh cuda_11.1.1_455.32.00_linux.run

# We have to manually download cuDNN 8, since the Conda version is 7.6.5, which is not compatible with CUDA 11
pip install gdown
gdown https://drive.google.com/uc?id=1heN3ax-Y5RxhDUKwH72PABxKLgBt7IT2

RUN conda install nccl -y

# 3) install packages
RUN pip install --no-cache-dir networkx scipy python-louvain

# Install pillow<7 due to dependency issue https://github.com/pytorch/vision/issues/1712
RUN pip install --no-cache-dir tensorflow-gpu \
								datascience \
								PyQt5 \
								scapy \
								nltk \
								opencv-contrib-python-headless \
								jupyter-tensorboard \
								opencv-python \
								pycocotools \

COPY ./kernels /usr/share/datahub/kernels
RUN conda env create --file /usr/share/datahub/kernels/ml-latest.yml && \
	conda init bash && \
	conda run -n ml-latest /bin/bash -c "pip install torch==1.5.0+cu101 torchvision==0.6.0+cu101 pytorch-ignite -f https://download.pytorch.org/whl/torch_stable.html; \
										 ipython kernel install --name=ml-latest"

RUN chown -R 1000:1000 /home/jovyan

COPY ./tests/ /usr/share/datahub/tests/scipy-ml-notebook
RUN chmod -R +x /usr/share/datahub/tests/scipy-ml-notebook


# 4) change back to notebook user
COPY /run_jupyter.sh /
USER $NB_UID

# Override command to disable running jupyter notebook at launch
# CMD ["/bin/bash"]
