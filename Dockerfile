FROM python:3.10-slim-bullseye
RUN pip3 install poetry
EXPOSE 7088
WORKDIR /opt/app
COPY data /opt/app/data
COPY pyproject.toml poetry.lock /opt/app/
RUN poetry install --only main --no-root
COPY src /opt/app/src
RUN poetry install --only-root
ENTRYPOINT ["poetry", "run", "python", "-m", "dedformer.api"]