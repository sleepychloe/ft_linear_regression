FROM	python:3.11

RUN	apt-get update \
		&& apt-get install -y \
			build-essential \
		&& rm -rf /var/lib/apt/lists/*

RUN	pip install --upgrade pip
RUN	pip install \
		numpy \
		pandas \
		matplotlib \
		jupyter

WORKDIR	/workspace

COPY	. .

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--no-browser", "--NotebookApp.token=''"]