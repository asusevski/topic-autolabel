# topic-autolabel

[![Documentation Status](https://readthedocs.org/projects/llama-cpp-python/badge/?version=latest)](https://llama-cpp-python.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-Apache-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://camo.githubusercontent.com/bb88127790fb054cba2caf3f3be2569c1b97bb45a44b47b52d738f8781a8ede4/68747470733a2f2f696d672e736869656c64732e696f2f656e64706f696e743f75726c3d68747470733a2f2f7261772e67697468756275736572636f6e74656e742e636f6d2f636861726c6965726d617273682f727566662f6d61696e2f6173736574732f62616467652f76312e6a736f6e)](https://github.com/charliermarsh/ruff)

Given text data, generates labels to classify the data into a set number of topics completely unsupervised.

## Example usage:

First, install the package with pip: ```pip install topic_autolabel```

```
# Labelling with supplied labels
from topic_autolabel import process_file
import pandas as pd

df = pd.read_csv('path/to/file')
candidate_labels = ["positive", "negative"]

# labelling column "review" with "positive" or "negative"
new_df = process_file(
    df=df,
    text_column="review",
    candidate_labels=candidate_labels,
    model_name="meta-llama/Llama-3.1-8B-Instruct" # default model to pull from huggingface hub
)
```

Alternatively, one can label text completely unsupervised by not providing the ```candidate_labels``` argument

```
from topic_autolabel import process_file
import pandas as pd

df = pd.read_csv('path/to/file')

# labelling column "review" with open-ended labels (best results when dataset talks about many topics)
new_df = process_file(
    df=df,
    text_column="review",
    model_name="meta-llama/Llama-3.1-8B-Instruct",
    num_labels=5 # generate up to 5 labels for each of the rows
)
```
