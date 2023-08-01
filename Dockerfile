FROM python:3.9

MAINTAINER Many Kasiriha <manykarim@users.noreply.github.com>
LABEL DocTest Library for Robot Framework in Docker

ARG release_name=gs952
ARG archive_name=ghostpcl-9.52-linux-x86_64

RUN pip install --no-cache-dir numpy
RUN pip install --no-cache-dir robotframework-doctestlibrary
WORKDIR    /
RUN apt-get update && apt-get install -y \
    imagemagick \
    tesseract-ocr \
    ghostscript \
    wget \
    libdmtx0b \
    software-properties-common \
    gettext-base \
    && rm -rf /var/lib/apt/lists/*

RUN wget -O /usr/share/tessdata/eng.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/eng.traineddata?raw=true \
    && wget -O /usr/share/tessdata/osd.traineddata https://github.com/tesseract-ocr/tessdata/blob/main/osd.traineddata?raw=true
          
RUN wget https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/${release_name}/${archive_name}.tgz \
  && tar -xvzf ${archive_name}.tgz \
  && chmod +x ${archive_name}/gpcl6* \
  && cp ${archive_name}/gpcl6* ${archive_name}/pcl6 \
  && cp ${archive_name}/* /usr/bin

COPY policy.xml /etc/ImageMagick-6/

WORKDIR    /