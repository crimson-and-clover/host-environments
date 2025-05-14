ARG BASE_IMAGE

FROM ${BASE_IMAGE}

ENV TZ="Asia/Shanghai"

ENV DEBIAN_FRONTEND="noninteractive"

COPY ./src/ /src

RUN bash -v /src/build.sh

CMD [ "bash", "/src/init.sh" ]

EXPOSE 22/tcp
