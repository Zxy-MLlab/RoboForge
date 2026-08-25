FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends bubblewrap git libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/roboforge
COPY pyproject.toml README.md ./
COPY embodied_codex ./embodied_codex
COPY evaluation ./evaluation
RUN python -m pip install --no-cache-dir .

RUN useradd --create-home --uid 10001 roboforge \
    && mkdir -p /runs /assets \
    && chown -R roboforge:roboforge /runs /assets
USER roboforge

ENV ROBOFORGE_ASSET_ROOT=/assets
ENTRYPOINT ["roboforge"]
CMD ["doctor", "--adapter", "embodied_codex.fake_adapter:FakeAdapter", "--model", "embodied_codex.fake_adapter:FakeModel", "--run-dir", "/runs/doctor"]
