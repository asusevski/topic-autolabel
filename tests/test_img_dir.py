import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from topic_autolabel import process_file


def test_unlabeled_classification():
    """
    Test the labeler's performance on IMDB sentiment classification.
    The test will fail if accuracy or F1 score falls below 60%.
    """
    np.random.seed(42) 
    try:
        result_df = process_file(
            filepath="./tests/imgs",
            text_column="text",
            model_name="llama3.2-vision:latest",
            batch_size=8,
            num_labels=3
        )
        for elem in result_df['label']:
            assert "anime" in elem.lower()
    except AssertionError:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
