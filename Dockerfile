# =============================================================================
# Dockerfile
# OpenMC + JupyterLab — OpenMC-Physics-informed Reactor-assembly optimization via Interchangeable AI Search Methods (openmc-prism)
#
# KEY DECISIONS from build history:
#   - ubuntu:22.04 fixed (not latest — unstable)
#   - OpenMC installed via shimwell pre-built wheels (not compiled from source)
#     saves ~30 min build time
#   - Nuclear data NOT baked into image — downloaded at container startup
#     via executedownload.sh (keeps image lean ~3 GB instead of ~11 GB)
#   - GPy removed — broken on Python 3.11
#   - h5py pinned to 3.10.0 — matches Ubuntu 22.04 libhdf5 (1.10.x)
#   - numpy pinned for 1.26.4 
#   - pv installed for extraction progress bar
#   - openmc_bo, openmc_ga and openmc_rl installed as packages via pyproject.toml
#   - environments/ is copied in as a plain importable folder (not pip
#     installed) — the notebooks add the repo root to sys.path and import
#     it directly, exactly like the local (non-Docker) workflow
#   - Notebooks/, environments/, openmc_bo/, openmc_ga/ ,openmc_rl/ and Results/ are all
#     copied in as siblings under /workspace so relative paths inside the
#     notebooks (e.g. "../Results/<model>", "../environments") resolve the
#     same way they do outside Docker
# =============================================================================

FROM ubuntu:22.04

# ── Suppress interactive apt prompts ─────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=UTC
ENV OPENMC_CROSS_SECTIONS=/nuclear_data/cross_sections.xml

# ── Build-time args ───────────────────────────────────────────────────────────
ARG PYTHON_VERSION=3.11

# ── System packages ───────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y \
        build-essential \
        git \
        wget \
        curl \
        pv \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-dev \
        python3-pip \
        libhdf5-dev \
        libpng-dev \
        libopenmpi-dev \
        openmpi-bin \
        gfortran \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default
RUN update-alternatives --install /usr/bin/python3 python3 \
        /usr/bin/python${PYTHON_VERSION} 1 \
    && update-alternatives --install /usr/bin/python python \
        /usr/bin/python${PYTHON_VERSION} 1

# ── Upgrade pip before installing requirements ───────────────────────────────
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# ── Python packages ───────────────────────────────────────────────────────────
# OpenMC is installed via shimwell's pre-built wheels index —
# no need to compile from source, saves ~30 min on every build.
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt \
    && rm /tmp/requirements.txt

# ── Install openmc_bo, openmc_ga and openmc_rl as Python packages ───────────────────────
# Copies each package's source so `from openmc_bo import ...` and
# `from openmc_ga import ...` work in any notebook without needing to
# manipulate sys.path manually.
COPY pyproject.toml /tmp/openmc_pkg/pyproject.toml
COPY openmc_bo/     /tmp/openmc_pkg/openmc_bo/
COPY openmc_ga/     /tmp/openmc_pkg/openmc_ga/
COPY openmc_rl/     /tmp/openmc_pkg/openmc_rl/
RUN pip3 install --no-cache-dir /tmp/openmc_pkg \
    && rm -rf /tmp/openmc_pkg

# ── Workspace ─────────────────────────────────────────────────────────────────
RUN mkdir -p /workspace/Notebooks /workspace/environments /workspace/Results
WORKDIR /workspace

# Notebooks — what JupyterLab opens into
COPY Notebooks/    /workspace/Notebooks/

# environments/ — model-specific OpenMC wrappers, imported directly by the
# notebooks via sys.path (see PROJECT_ROOT = os.path.abspath("..")).
COPY environments/  /workspace/environments/

# Results/ — pre-existing sample results, plus the directory the notebooks
# write new runs into.
COPY Results/       /workspace/Results/

# ── JupyterLab configuration ──────────────────────────────────────────────────
RUN jupyter lab --generate-config \
    && printf '%s\n' \
        "c.ServerApp.ip = '0.0.0.0'" \
        "c.ServerApp.port = 8888" \
        "c.ServerApp.open_browser = False" \
        "c.ServerApp.token = ''" \
        "c.ServerApp.password = ''" \
        "c.ServerApp.allow_root = True" \
        "c.ServerApp.notebook_dir = '/workspace'" \
       >> /root/.jupyter/jupyter_lab_config.py

# ── Entrypoint ────────────────────────────────────────────────────────────────
# executedownload.sh checks for nuclear data on startup,
# downloads if missing, then starts JupyterLab.
COPY executedownload.sh /executedownload.sh
RUN chmod +x /executedownload.sh

EXPOSE 8888

ENTRYPOINT ["/executedownload.sh"]
