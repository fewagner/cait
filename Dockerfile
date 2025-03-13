ARG FLAVOUR=-slim
FROM gitlab-registry.cern.ch/cryocluster/python-container-prebuild:jupyter_hub_base_image

ENV VIRTUAL_ENV=/opt/venv_container
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
ENV JUPYTER_PREFER_ENV_PATH=1

# Copy cait repository
COPY . /opt/programs/cait

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install cait (including optional dependencies)
# The 'nn' optional dependency (including torch) is not installed in the 'slim' version
RUN echo $FLAVOUR
RUN if [ "$FLAVOUR" = "-slim" ] ; then \
        echo "Building slim container"; \
        python -m pip install -e /opt/programs/cait[clplot,remfiles]; \
    else \
        echo "Building full container"; \
        python -m pip install -e /opt/programs/cait[nn,clplot,remfiles]; \
    fi
