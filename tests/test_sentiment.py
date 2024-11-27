import pytest
from datasets import load_dataset
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from topic_autolabel import process_file
import tempfile
import os


def test_sentiment_classification():
    """
    Test the labeler's performance on IMDB sentiment classification.
    The test will fail if accuracy or F1 score falls below 60%.
    """
    dataset = load_dataset("stanfordnlp/imdb", split="test")
    
    df = pd.DataFrame(dataset).sample(n=200, random_state=42)
    
    df['label'] = df['label'].map({1: "positive", 0: "negative"})
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
        accuracy = accuracy_score(df['label'], result_df['label'])
        f1 = f1_score(df['label'], result_df['label'], pos_label="positive")
        print(f"Accuracy: {accuracy:.2%}")
        print(f"F1 Score: {f1:.2%}")
        
        assert accuracy >= 0.60, f"Accuracy {accuracy:.2%} below threshold of 60%"
        assert f1 >= 0.60, f"F1 Score {f1:.2%} below threshold of 60%"
        
    finally:
        os.unlink(temp_filepath)


if __name__ == "__main__":
    pytest.main([__file__])