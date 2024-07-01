#!/bin/bash

DIR="$(pwd)/miniconda3"

install_conda()
{
	curl -o Miniconda3-latest-Linux-x86_64.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
	chmod +x Miniconda3-latest-Linux-x86_64.sh
	sh Miniconda3-latest-Linux-x86_64.sh -u -b -p "$DIR"
}

setup_env()
{
	source "$DIR/etc/profile.d/conda.sh"
	"$DIR/bin/conda" config --set auto_activate_base false
	"$DIR/bin/conda" create -n myenv python=3.8 matplotlib pandas -y
	conda activate myenv
	conda config --add channels conda-forge
	conda install -n myenv gcc libstdcxx-ng -y
	export LD_LIBRARY_PATH="$DIR/envs/myenv/lib:/usr/lib/x86_64-linux-gnu/dri:$LD_LIBRARY_PATH"
	export LIBGL_DEBUG=verbose
	export LIBGL_ALWAYS_SOFTWARE=1
	export QT_QPA_PLATFORM=xcb
	export MESA_LOADER_DRIVER_OVERRIDE=swrast
}

if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
	install_conda
	if [ -d "$DIR" ]; then
		setup_env
	fi
fi
