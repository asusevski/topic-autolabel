# topic-autolabel
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

## labelling column "review" with "positive" or "negative"
new_df = process_file(
    filepath=temp_filepath,
    text_column="review",
    candidate_labels=candidate_labels
)
```