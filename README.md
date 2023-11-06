This repository contains the code for the dual encoder model described in "Code Search as Multi-Link Translation Using Dual Encoders".


## Dependencies
The required dependencies must be installed to run the source code.
```
pip install -r requirements.txt
```

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
Trained FastText word embeddings for the datasets' can be found [here](https://drive.google.com/drive/folders/19IjAwyswD8PRmwZTuyU0yVe28zbWTKYg?usp=drive_link). Download the files and place them inside the Data/cbow/CodeSearch300 directory. These embeddings have a dimension size of 300, trained on a FastText CBOW model.

It is possible to train a different FastText model or with a different dimension size. The raw text data for that purpose can be found [here](https://drive.google.com/drive/folders/1ymBCRS25LSku5QqfUtmQXlZZC7YdmCXa?usp=sharing). The steps for training a FastText model in this data are as follows:-

1. Download the raw text data and place the files in the Data/Texts directory.
2. If none of the dependencies are installed, install the FastText and Pandas packages.
```
pip install fasttext
pip install pandas
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
To run experiments with separate language models, run the ProcessDataSeparate.py file.
```
python ProcessDataSeparate.py
```

## Code search
Once the dependencies are installed, run the corresponding files to run the code search experiment on that dataset with the default parameters. The parameters values can be changed to conduct different experiments.

For CodeSearchNet Python (Limited), run -
```
python CodeSearch.py
```
For CodeSearchNet Python (Full), run -
```
python CodeSearchFull.py
```
For AdvTest Python, run -
```
python AdvTest.py
```
For DGMS, run -
```
python DGMS.py
```
