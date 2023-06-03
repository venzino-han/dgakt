default: build

help:
	@echo 'Management commands for dgkt:'
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
	@docker build . -t dgkt 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name dgkt -v `pwd`:/workspace/dgkt dgkt:latest /bin/bash

up: build run

rm: 
	@docker rm dgkt

stop:
	@docker stop dgkt

reset: stop rm
