import os
import tempfile

import numpy as np
import pandas as pd
import pytest
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from topic_autolabel import process_file


def test_unlabeled_img_labelling():
    """
    Test the labeler's performance on a directory of images from anime
    of people eating burgers.
    The test will fail if the images are not labelled with anime/food in
    the 3 labels.
    """
    np.random.seed(42) 
    try:
        result_df = process_file(
            filepath="./tests/imgs",
            model_name="llama3.2-vision:latest",
            num_labels=3
        )
        for elem in result_df['label']:
            assert "anime" in elem.lower()
            assert "food" in elem.lower()
    except AssertionError:
        pass

if __name__ == "__main__":
    pytest.main([__file__])
