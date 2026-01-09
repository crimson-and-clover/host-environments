ARG BASE_IMAGE

FROM ${BASE_IMAGE}

# Expose SSH port
EXPOSE 22/tcp

COPY ./src/ /src

RUN bash -v /src/build.sh

CMD [ "bash", "/src/init.sh" ]
