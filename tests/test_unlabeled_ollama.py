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
    df = pd.DataFrame({'text': np.random.choice(['this is a color', 'this is a food', 'this is a movie'], size=100)})
    def parse_label(text: str) -> int:
        if text == "this is a color":
            return 0
        elif text == "this is a food":
            return 1
        else:
            return 2
    df['ground_truth'] = df['text'].apply(parse_label)
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_filepath = f.name
    try:
        result_df = process_file(
            model_name="llama3.1",
            filepath=temp_filepath,
            text_column="text",
	        batch_size=8,
            num_labels=3
        )
        result_df['label'] = result_df['label']
        y_true = result_df['ground_truth']
        def parse_pred_label(text: str) -> int:
            if text.lower() == "color":
                return 0
            elif text.lower() == "food":
                return 1
            else:
                return 2
        y_pred = result_df['label'].apply(parse_pred_label)
        accuracy = accuracy_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2%}")
        assert accuracy >= 0.80, f"Accuracy {accuracy:.2%} below threshold of 60%"
    finally:
        os.unlink(temp_filepath)


if __name__ == "__main__":
    pytest.main([__file__])