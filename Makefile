# **************************************************************************** #
#                                                                              #
#                                                         :::      ::::::::    #
#    Makefile                                           :+:      :+:    :+:    #
#                                                     +:+ +:+         +:+      #
#    By: yhwang <yhwang@student.42.fr>              +#+  +:+       +#+         #
#                                                 +#+#+#+#+#+   +#+            #
#    Created: 2024/06/26 19:02:53 by yhwang            #+#    #+#              #
#    Updated: 2024/06/26 20:25:53 by yhwang           ###   ########.fr        #
#                                                                              #
# **************************************************************************** #

RESET		:= $(shell tput -Txterm sgr0)
YELLOW		:= $(shell tput -Txterm setaf 3)

SHELL := /bin/bash

all: setup run

setup:
ifeq ($(shell ls | grep Miniconda3-latest-Linux-x86_64.sh | wc -l), 0)
	@echo "$(YELLOW) Miniconda3-latest-Linux-x86_64.sh not found$(RESET)"
	@echo "$(YELLOW) Running install_conda.sh$(RESET)"
	./install_conda.sh
else
	@echo "$(YELLOW) Miniconda3-latest-Linux-x86_64.sh found$(RESET)"
endif

run:
	source $(shell pwd)/miniconda3/bin/activate myenv \
		&& $(shell pwd)/miniconda3/envs/myenv/bin/python ./srcs/ex.py

fclean:
	source $(shell pwd)/miniconda3/etc/profile.d/conda.sh \
		&& conda env remove --name myenv --yes
	rm -rf miniconda3
	rm -rf Miniconda3-latest-Linux-x86_64.sh

.PHONY: all setup run fclean
