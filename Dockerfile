FROM gitlab-registry.cern.ch/cryocluster/python-container-prebuild:jupyter_hub_base_image

ENV VIRTUAL_ENV=/opt/venv_container
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV JUPYTER_PREFER_ENV_PATH=1

# copy cait repository
COPY . /opt/programs/cait

# upgrade pip and install cait (including optional dependencies)
RUN python -m pip install --upgrade pip \
    && python -m pip install -e /opt/programs/cait[nn,clplot,remfiles]