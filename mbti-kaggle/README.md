# mbti-kaggle

Our take on the [MBTI dataset uploaded on Kaggle](https://www.kaggle.com/datasnaek/mbti-type),  
using the skillset learned from the online version of [Stanford's CS224n](http://web.stanford.edu/class/cs224n/).


## Setup

This repository was run in Python 3.8.  
Dependencies can be installed via pip:
```python
pip install -r requirements.txt
```


## Experimental Results

### Multiclass Classification

Classification accuracy and F1 score under 3-fold cross validation (single seed)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.6778        | 0.6665        |
| Original  | CountVectorizer    | MLP          | 0.6016        | 0.5747        |
| Original  | LanguageModel      | MLP          | **0.7037**    | **0.6939**    |
| Masked    | CountVectorizer    | Classical ML | 0.4854        | 0.4476        |
| Masked    | CountVectorizer    | MLP          | 0.4360        | 0.4058        |
| Masked    | LanguageModel      | MLP          | **0.4958**    | **0.4718**    |
| Hypertext | CountVectorizer    | Classical ML | 0.4889        | 0.4508        |
| Hypertext | CountVectorizer    | MLP          | 0.4432        | 0.4107        |
| Hypertext | LanguageModel      | MLP          | **0.4946**    | **0.4709**    |


### Binary Classification

Classification accuracy and F1 score under 3-fold cross validation (single seed)  
arranged below in the order of the MBTI functions


#### Extraversion (E) vs. Introversion (I)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8587        | 0.8489        |
| Original  | CountVectorizer    | MLP          | 0.8303        | 0.8119        |
| Original  | LanguageModel      | MLP          | **0.8876**    | **0.8800**    |
| Masked    | CountVectorizer    | Classical ML | 0.8038        | 0.7747        |
| Masked    | CountVectorizer    | MLP          | 0.7909        | 0.7660        |
| Masked    | LanguageModel      | MLP          | **0.8146**    | **0.7904**    |
| Hypertext | CountVectorizer    | Classical ML | 0.8034        | 0.7745        |
| Hypertext | CountVectorizer    | MLP          | 0.7913        | 0.7664        |
| Hypertext | LanguageModel      | MLP          | **0.8161**    | **0.7937**    |


#### Sensing (S) vs. Intuition (N)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.9132        | 0.9050        |
| Original  | CountVectorizer    | MLP          | 0.8924        | 0.8720        |
| Original  | LanguageModel      | MLP          | **0.9251**    | **0.9148**    |
| Masked    | CountVectorizer    | Classical ML | 0.8660        | 0.8193        |
| Masked    | CountVectorizer    | MLP          | 0.8625        | 0.8313        |
| Masked    | LanguageModel      | MLP          | **0.8719**    | **0.8557**    |
| Hypertext | CountVectorizer    | Classical ML | 0.8644        | 0.8178        |
| Hypertext | CountVectorizer    | MLP          | 0.8610        | 0.8290        |
| Hypertext | LanguageModel      | MLP          | **0.8739**    | **0.8581**    |


#### Thinking (T) vs. Feeling (F)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8583        | **0.8583**    |
| Original  | CountVectorizer    | MLP          | 0.8383        | 0.8389        |
| Original  | LanguageModel      | MLP          | **0.8589**    | 0.8557        |
| Masked    | CountVectorizer    | Classical ML | **0.7899**    | **0.7898**    |
| Masked    | CountVectorizer    | MLP          | 0.7818        | 0.7827        |
| Masked    | LanguageModel      | MLP          | 0.7703        | 0.7622        |
| Hypertext | CountVectorizer    | Classical ML | **0.7957**    | **0.7956**    |
| Hypertext | CountVectorizer    | MLP          | 0.7746        | 0.7756        |
| Hypertext | LanguageModel      | MLP          | 0.7756        | 0.7706        |


#### Judging (J) vs. Perceiving (P)

| Preprocessing | Vectorization | Classifier | Accuracy | F1 | 
|---|---|---|---|---|
| Original  | CountVectorizer    | Classical ML | 0.8026        | 0.8008        |
| Original  | CountVectorizer    | MLP          | 0.7530        | 0.7501        |
| Original  | LanguageModel      | MLP          | **0.8524**    | **0.8494**    |
| Masked    | CountVectorizer    | Classical ML | 0.7197        | 0.7104        |
| Masked    | CountVectorizer    | MLP          | 0.6898        | 0.6848        |
| Masked    | LanguageModel      | MLP          | **0.7274**    | **0.7179**    |
| Hypertext | CountVectorizer    | Classical ML | 0.7135        | 0.7047        |
| Hypertext | CountVectorizer    | MLP          | 0.6830        | 0.6772        |
| Hypertext | LanguageModel      | MLP          | **0.7320**    | **0.7224**    |


## Usage

The commands for reproducing the results for multiclass classification are shown below.

Original + CountVectorizer + Classical ML
```python
python main.py --dataset kaggle \
               --loader CountVectorizer \
               --method ensemble \
               --n_splits 3 \
               --seed 100
```

Masked + CountVectorizer + MLP
```python
python main.py --dataset kaggle_masked \
               --loader CountVectorizer \
               --method sgd \
               --model mlp3 \
               --batch_size 16 \
               --lr 2e-5 \
               --epochs 10 \
               --dropout 0.1 \
               --bn \
               --n_splits 3 \
               --seed 100
```

Hypertext + LanguageModel + MLP  
Note that the required vram is about 20Gb, due to the length of the input sequence.
```python
python main.py --dataset hypertext \
               --loader LanguageModel \
               --method sgd \
               --model lm_classifier \
               --lm distilgpt2 \
               --max_length 1024 \
               --batch_size 8 \
               --lr 2e-5 \
               --epochs 5 \
               --n_splits 3 \
               --seed 100
```
