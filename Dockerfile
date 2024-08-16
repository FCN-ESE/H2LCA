# Specify parent image
ARG BASE_IMAGE=registry.git.rwth-aachen.de/jupyter/profiles/rwth-minimal
FROM ${BASE_IMAGE}

# Update conda base environment to match specifications in environment.yml
ADD environment.yml /tmp/environment.yml
USER root
RUN sed -i "s|name\: tesa|name\: base|g" /tmp/environment.yml # we need to replace the name of the environment with base such that we can update the base environment here
USER $NB_USER

# All packages specified in environment.yml are installed in the base environment
RUN conda env update -f /tmp/environment.yml && \
	conda clean --all -f -y

RUN jupyter labextension install @jupyter-widgets/jupyterlab-manager jupyter-matplotlib

# Download and install glpk
USER root
RUN apt-get update
RUN apt-get install -y build-essential
RUN mkdir /tmp/glpk \
	&& wget -O- https://ftp.gnu.org/gnu/glpk/glpk-4.65.tar.gz \
	| tar xvzC /tmp/glpk --strip-components=1 \
	&& cd /tmp/glpk \
	&& ./configure \
	&& make && make install \
	&& rm -rf /tmp/glpk

RUN echo '/usr/local/lib' > /etc/ld.so.conf.d/local.conf
RUN ldconfig
USER $NB_USER
