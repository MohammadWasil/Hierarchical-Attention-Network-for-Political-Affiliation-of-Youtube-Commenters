## Training Description
To train and evaluate your model:

We will train 2 models: HAN and LSTM.


### Dependencies:
##### Version:
```
Pytroch version : #.#.#+cu##
```

##### Steps
1. Create an environment <Br/>
```conda create -n pytorch python=3.7```

2. Activate the environment <Br/>
```conda activate pytorch```

3. Install jupyter<Br/>
```conda install -c anaconda jupyter```

4. Install the ipykernel<Br/>
```pip install ipykernel```

5. Register your environment<Br/>
```python -m ipykernel install --user --name pytorch --display-name "pytorch"```

6. Install [pytorch](https://pytorch.org/get-started/locally/)<Br/>
GPU Version: ```conda install pytorch cudatoolkit -c pytorch``` <Br/>
CPU Version: ```conda install pytorch cpuonly -c pytorch```

### Train
#### Train HAN model
Initially, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train HAN model:
```{bash}
python run.py --MODEL HAN --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

To test your HAN model, run:
```{bash}
python run.py --MODEL HAN --RUN_MODE test
```

#### Train LSTM model
Initially, `LAST_SAVED_EPOCH_LSTM_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train LSTM model:
```{bash}
python run.py --MODEL LSTM --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

To test your LSTM model, run:
```{bash}
python run.py --MODEL LSTM --RUN_MODE test
```
