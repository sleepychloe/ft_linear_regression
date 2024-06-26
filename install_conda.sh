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
	"$DIR"/bin/conda config --set auto_activate_base false
	conda create -n myenv python=3.8 matplotlib pandas -y
	source "$DIR/bin/activate" myenv
	export LD_LIBRARY_PATH="$DIR/envs/myenv/lib:$LD_LIBRARY_PATH"
	export LIBGL_ALWAYS_SOFTWARE=1
	export QT_QPA_PLATFORM=xcb
}

if [ ! -f "Miniconda3-latest-Linux-x86_64.sh" ]; then
	install_conda
	if [ -d "$DIR" ]; then
		setup_env
	fi
fi
