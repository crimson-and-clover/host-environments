ARG BASE_IMAGE

FROM ${BASE_IMAGE}

COPY ./src/ /root/src

RUN apt-get update && apt-get upgrade -y

RUN apt-get install -y openssh-server build-essential sudo git cmake curl wget net-tools vim zip ffmpeg

RUN gcc /root/src/pause.c -o /usr/bin/pause && nvcc /root/src/cuda_occupier.cu -o /usr/bin/cuda_occupier

RUN apt-get install -y libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6

RUN wget -O /root/Anaconda3-Linux-x86_64.sh https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh && \
    bash /root/Anaconda3-Linux-x86_64.sh -b -p /root/anaconda3 && \
    /root/anaconda3/bin/conda init

RUN mkdir -p /root/.ssh && \
    chmod -R 700 /root/.ssh && \
    cp -a /root/src/authorized_keys /root/.ssh/ && \
    chmod 644 /root/.ssh/authorized_keys

RUN sed -i '/#PermitRootLogin/s/.*/PermitRootLogin yes/' /etc/ssh/sshd_config

EXPOSE 22/tcp

RUN echo "root:000000" | chpasswd

CMD [ "bash", "/root/src/init.sh" ]
