FROM python:3.12-slim

LABEL org.opencontainers.image.authors="Many Kasiriha <manykarim@users.noreply.github.com>"
LABEL org.opencontainers.image.description="DocTest Library for Robot Framework in Docker"

ARG release_name=gs9561
ARG archive_name=ghostpcl-9.56.1-linux-x86_64
# Install source: the checked-out code by default (CI tests the PR's code);
# pass --build-arg INSTALL_SOURCE='robotframework-doctestlibrary[ai]' for a
# release image from PyPI.
ARG INSTALL_SOURCE=/src[ai]

WORKDIR /

RUN apt-get update && apt-get install -y \
  imagemagick \
  tesseract-ocr \
  ghostscript \
  wget \
  libdmtx0b \
  libzbar0 \
  gettext-base \
  && rm -rf /var/lib/apt/lists/*

RUN wget https://github.com/ArtifexSoftware/ghostpdl-downloads/releases/download/${release_name}/${archive_name}.tgz \
  && tar -xvzf ${archive_name}.tgz \
  && chmod +x ${archive_name}/gpcl6* \
  && cp ${archive_name}/gpcl6* ${archive_name}/pcl6 \
  && cp ${archive_name}/* /usr/bin \
  && rm -rf ${archive_name} ${archive_name}.tgz

COPY policy.xml /etc/ImageMagick-6/

# Copy only what the build needs (never the whole context — it can
# contain local secrets like .env), so CI images test the PR's code
# instead of the last released package.
COPY pyproject.toml README.md /src/
COPY DocTest /src/DocTest
COPY doctest_dashboard /src/doctest_dashboard
RUN pip install --no-cache-dir "$INSTALL_SOURCE" && rm -rf /src

WORKDIR /
