import os
import tempfile

import pandas as pd
import pytest
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score

from topic_autolabel import process_file


def test_sentiment_classification():
    """
    Test the labeler's performance on IMDB sentiment classification.
    The test will fail if accuracy or F1 score falls below 60%.
    """
    dataset = load_dataset("stanfordnlp/imdb", split="test")
    
    df = pd.DataFrame(dataset).sample(n=200, random_state=42)
    
    df['label'] = df['label'].apply(lambda x: "positive" if x == 1 else "negative")
    df = df.rename(columns={"text": "review"})
    
    with tempfile.NamedTemporaryFile(suffix='.csv', mode='w', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_filepath = f.name
    
    try:
        candidate_labels = ["positive", "negative"]
        result_df = process_file(
            filepath=temp_filepath,
            text_column="review",
            candidate_labels=candidate_labels
        )
        result_df['label'] = result_df['label'].replace("<err>", candidate_labels[0])
        y_true = (df['label'] == "positive").astype(int)
        y_pred = (result_df['label'] == "positive").astype(int)
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.2%}")
        
        assert accuracy >= 0.60, f"Accuracy {accuracy:.2%} below threshold of 60%"
        assert f1 >= 0.60, f"F1 Score {f1:.2%} below threshold of 60%"
        
    finally:
        os.unlink(temp_filepath)


if __name__ == "__main__":
    pytest.main([__file__])