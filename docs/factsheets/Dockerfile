FROM ghcr.io/quarto-dev/quarto:1.7.19 AS build_docs

RUN mkdir /quarto-run /quarto-extensions

WORKDIR /quarto-extensions

RUN quarto install extension --no-prompt debruine/quarto-glossary

WORKDIR /quarto-run

COPY build.sh .

CMD /bin/bash -c './build.sh; /bin/bash'