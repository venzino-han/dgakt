default: build

help:
	@echo 'Management commands for dgakt:'
	@echo
	@echo 'Usage:'
	@echo '    make build          Build image'
	@echo '    make run            Run image'
	@echo '    make up             Build and run image'
	@echo '    make reset          Stop and remove container'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t dgakt 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name dgakt -v `pwd`:/workspace/dgakt dgakt:latest /bin/bash

up: build run

rm: 
	@docker rm dgakt

stop:
	@docker stop dgakt

reset: stop rm
