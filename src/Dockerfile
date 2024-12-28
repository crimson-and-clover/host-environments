ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV TZ="Asia/Shanghai"

ENV DEBIAN_FRONTEND="noninteractive"

COPY ./src/ /src

RUN bash -v /src/build.sh

CMD [ "bash", "/src/init.sh" ]

EXPOSE 22/tcp

VOLUME [ "/home/developer/anaconda3/envs", "/home/developer/anaconda3/pkgs", "/home/developer/.vscode-server" ]
VOLUME [ "/home/developer/.cache/huggingface", "/home/developer/.cache/pip" ]
VOLUME [ "/home/developer/hdd", "/home/developer/ssd" ]
