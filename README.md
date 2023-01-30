This repository contains the code for the dual encoder model described in "Code Search as Multi-Link Translation Using Dual Encoders".

## Directory structure
The following commands must be run to build the directory structure for the experiments.
```
mkdir -p Data/{cbow,Texts,Trained_models}
mkdir Data/cbow/CodeSearch300
mkdir Models
mkdir Plots
mkdir Results/{CodeSearch,Tensors}
```

## Data
Trained FastText word embeddings for the CodeSearchNet dataset's six divisions can be found [here](https://drive.google.com/file/d/1z--18A12T6JEBO5q8-q_MwB0lCGN-gaN/view?usp=sharing). Download the files and place them inside the Data/cbow/CodeSearch300 directory. These embeddings have a dimension size of 300, trained on a FastText CBOW model.

It is possible to train a different FastText model or with a different dimension size. The raw text data for that purpose can be found [here](https://drive.google.com/file/d/11HBX-D7Y7E8Hjhx6OZoZ8v4LuokLJDYV/view?usp=sharing). The steps for training a FastText model in this data are as follows:-

1. Download the raw text data and place the files in the Data/Texts directory.
2. Install the FastText package
```
pip install fasttext
```
3. Edit the ``embedding`` and ``dimensions`` variables' values in ProcessData.py to desired values. The default values are -
```
embedding = "cbow"
dimensions = 300
```
When the dimension size is not specified in ``fasttext.train_unsupervised()``, the embeddings have a default size of 100.

4. After assigning desired values to these parameters, run the ProcessData.py file.
```
python ProcessData.py
```

## Code search
To run the code search experiment with the default parameters, run the CodeSearch.py file.
```
python CodeSearch.py
```

The parameters values can be changed to conduct different experiments. The parameters and their values are explained in CodeSearch.py.
