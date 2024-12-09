# topic-autolabel

[![Documentation Status](https://readthedocs.org/projects/llama-cpp-python/badge/?version=latest)](https://llama-cpp-python.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org)

Given text data, generates labels to classify the data into a set number of topics completely unsupervised.

---
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
