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

### HAN model
#### Train

Initially, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train HAN model:
```{bash}
python run.py --MODEL HAN --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name. And also make sure to change `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

#### Test
To test your HAN model, run:
```{bash}
python run.py --MODEL HAN --RUN_MODE test
```

#### Inference
For inferencing your model on un-labeled data, make sure to change `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` with your best model, `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase. This will gives us results of number of liberals and conservatives, and also creates a pickle file containing authors id, and their biasness. We had 2 files to infernced, and hence we got 2 dictionary file. We ran this cmd for the 2 files separately. Then, we ran the Script "Scripts/18. remove_conflicts_inference.py" to remove the conflicts and merge the result.
```{bash}
python run.py --MODEL HAN --RUN_MODE inference
```

#### Visualize
To visualize our HAN model, this creates an html file containing the visualizations of attention score the HAN model gives to each words and each sentences in a document. Check the html file attached on the notebook.
```{bash}
python run.py --MODEL HAN --RUN_MODE vis
```

### LSTM model
#### Train
Initially, `LAST_SAVED_EPOCH_LSTM_MODEL` parameter in `config.yml` file will be `Null`. At this point, simply run the below command to train LSTM model:
```{bash}
python run.py --MODEL LSTM --RUN_MODE train
```

To resume training after some time, `LAST_SAVED_EPOCH_HAN_MODEL` parameter in `config.yml` needs to be changed to the last epoch trained. For that, check the epoch number in the saved model's file name. And also make sure to change `PATH_TO_SAVE_MODEL_HAN` with the path of your best model, and `PATH_TO_SAVE_VOCAB_HAN` with the vocababulary created during training phase.
For that, go to `./Han Models/`. And then, run the above command again to resume training.

#### Test
To test your LSTM model, run:
```{bash}
python run.py --MODEL LSTM --RUN_MODE test
```

No inference script was created for LSTM, since we already know HAN was working much better than LSTM model.
