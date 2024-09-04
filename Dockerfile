# syntax=docker/dockerfile:1.2

FROM public.ecr.aws/lambda/python:3.9 as base

# Poetry
RUN curl -sSL https://install.python-poetry.org | python3 - --version 1.1.12
ENV PATH /root/.local/bin:$PATH

# Config
COPY poetry.lock pyproject.toml ./

# Dependencies
RUN poetry config virtualenvs.create false
RUN poetry install --no-dev --no-interaction --no-ansi

# Models
RUN mkdir -p ./models/
COPY ./models/ ./models/

# Config
COPY .env cfg_cloud.env ./
ENV PYTHONPATH "${PYTHONPATH}:/var/lang/bin/python3.9"

# Code
RUN mkdir ./src
COPY ./src ./src

# Lambda
CMD ["src.app.api.handler"]