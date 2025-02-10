FROM python:3.11-slim
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

RUN apt-get update -y && \
    apt-get -y --no-install-recommends install git unzip build-essential && \
    rm -rf /var/lib/apt/lists/*

# Add repository code
WORKDIR /opt/app
COPY src /opt/app/src
COPY pyproject.toml /opt/app
COPY .git /opt/app/.git

RUN uv sync --no-dev

ENTRYPOINT ["uv", "run", "--no-dev"]
CMD ["cloudcasting-app"]
