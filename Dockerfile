FROM ubuntu:18.04 
 
RUN apt update --fix-missing 
RUN apt-get install -y software-properties-common 
RUN add-apt-repository ppa:deadsnakes/ppa 
RUN apt update --fix-missing 
RUN apt-get install -y ffmpeg python3.6 python3.6-dev openssh-server libxrender1 libsm6 wget gcc iputils-ping  curl vim nginx
 
RUN apt-get install -y python3-distutils && apt-get install -y python3-apt 
RUN apt install -y libgtk-3-0 && rm -rf /var/lib/apt/lists/* 
 

# WORKDIR /app
COPY get-pip.py . 
RUN python3.6 get-pip.py 
RUN ln -s /usr/bin/python3.6 /usr/local/bin/python3 
 
COPY requirement.txt . 
RUN pip3 install -r requirement.txt 
 
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/program:${PATH}"

COPY yolov4 /opt/program
WORKDIR /opt/program
 