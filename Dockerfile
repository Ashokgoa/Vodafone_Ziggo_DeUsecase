# Slim Debian-based image, not Alpine: our heaviest dependencies (torch,
# chromadb's onnxruntime, numpy, scipy) ship pre-built Linux wheels for
# standard glibc, but not for Alpine's musl libc -- Alpine would force slow
# from-source builds or outright failures for these packages. "slim" is the
# small-but-compatible middle ground.
FROM python:3.11-slim

WORKDIR /app

# Copy the project and install it. For a project this size we accept that
# any source change invalidates this layer's build cache (a fresh
# `pip install .`); splitting out a separate requirements.txt purely for
# caching would duplicate pyproject.toml's dependency list for marginal
# benefit here. README.md is required too: pyproject.toml references it
# as the package readme, and setuptools reads it during the build.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir .

EXPOSE 8000

# --host 0.0.0.0 is required inside a container: the default 127.0.0.1
# only accepts connections from inside the container itself, which would
# make the API unreachable from the host machine (and from Compose's own
# healthcheck).
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
