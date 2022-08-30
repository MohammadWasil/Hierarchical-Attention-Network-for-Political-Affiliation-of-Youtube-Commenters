## Training Description

A lot of help, inspiration and code for HAN model was taken from these sources.

[Hierarchical Attention Networks | Lecture 50 (Part 2) | Applied Deep Learning - Maziar Raissi](https://www.youtube.com/watch?v=VBqbmmcMI7E&ab_channel=MaziarRaissi)<Br>
[Predicting Amazon review scores using Hierarchical Attention Networks with PyTorch and Apache Mxnet](https://towardsdatascience.com/predicting-amazon-reviews-scores-using-hierarchical-attention-networks-with-pytorch-and-apache-5214edb3df20) <Br>
[[PYTORCH] Hierarchical Attention Networks for Document Classification - uvipen](https://github.com/uvipen/Hierarchical-attention-networks-pytorch) <Br>
[Text Classification with Hierarchical Attention Networks](https://humboldt-wi.github.io/blog/research/information_systems_1819/group5_han/) <Br>
[Pytorch-Hierarchical-Attention-Network - JoungheeKim](https://github.com/JoungheeKim/Pytorch-Hierarchical-Attention-Network) <Br>
[Hierarchical-Attention-Network - jaehunjung1](https://github.com/jaehunjung1/Hierarchical-Attention-Network) <Br>

### Dependencies:
##### Version:
```
Pytroch version :  1.10.0+cu102
torchtext version : 0.11.0
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
We will train 2 models: HAN and LSTM.
  
#### HAN model
Initially, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train HAN model:
```{bash}
python run.py --MODEL HAN --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name. And also make sure to change `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

To test your HAN model, run:
```{bash}
python run.py --MODEL HAN --RUN_MODE test
```

To Evaluate yout model, make sure to change `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` with your best model, `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase.
```{bash}
python run.py --MODEL HAN --RUN_MODE eval
```

#### LSTM model
Initially, `LAST_SAVED_EPOCH_LSTM_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train LSTM model:
```{bash}
python run.py --MODEL LSTM --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name. And also make sure to change `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

To test your LSTM model, run:
```{bash}
python run.py --MODEL LSTM --RUN_MODE test
```

No evaluation script was created for LSTM, since we already know HAN was working much better than LSTM model.
