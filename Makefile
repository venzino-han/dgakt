default: build

help:
	@echo 'Management commands for dagkt:'
	@echo
	@echo 'Usage:'
	@echo '    make build            Build image'
	@echo '    make pip-sync         Pip sync.'

preprocess:
	@docker 

build:
	@echo "Building Docker image"
	@docker build . -t dagkt 

run:
	@echo "Booting up Docker Container"
	@docker run -it --gpus '"device=0"' --ipc=host --name dagkt -v `pwd`:/workspace/dagkt dagkt:latest /bin/bash

up: build run

rm: 
	@docker rm dagkt

stop:
	@docker stop dagkt

reset: stop rm
