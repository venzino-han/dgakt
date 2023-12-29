# Dual Graph Attention-based Knowledge Tracing

# How to run 


## Docker Container
- Docker container use DGAKT project directory as volume 
- File change will be apply directly to file in docker container

## Download datasets
1. `make up` : build docker image and start docker container
2. `python3 download_datasets.py` : download datasets

## Preprocessing
1. `make up` : build docker image and start docker container
2. `python3 src/preprocess.py` : start data preprocessing in docker container
```
# EdNet
python3 src/preprocess.py -d ednet -s ednet

# ASSIST2017 
python3 src/preprocess.py -d assist -s assist

# Junyi
python3 src/preprocess.py -d junyi -s junyi
```
3. `python3 src/preprocess_part.py` : start data preprocessing for unseen case

## Evaluate
1. `make up` : build docker image and start docker container
2. check `test_config/test_list.ymal` file
3. `python3 src/evaluate.py` : start train in docker container

## Train 
1. `make up` : build docker image and start docker container
2. check `train_config/train_list.ymal` file
3. `python3 src/train.py` : start train in docker container

<br />