from typing import List, Optional

import pandas as pd

from .core.data_loader import load_data
from .core.labeler import TopicLabeler


def process_file(
    filepath: str,
    text_column: str,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_labels: int = 5,
    candidate_labels: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Process a file and add topic labels to it.

    Args:
        filepath: Path to the CSV file
        text_column: Name of the column containing text to process
        model_name: Name of the HuggingFace model to use
        num_labels: Number of labels to generate (if candidate_labels is None)
        candidate_labels: List of predefined labels to choose from (optional)

    Returns:
        DataFrame with a new 'label' column containing the generated labels
    """
    # Load the data
    df = load_data(filepath, text_column)

    # Initialize the labeler
    labeler = TopicLabeler(model_name=model_name)

    # Generate labels
    labels = labeler.generate_labels(
        df[text_column].tolist(),
        num_labels=num_labels,
        candidate_labels=candidate_labels,
    )

    # Add labels to dataframe
    df["label"] = labels

    return df


__all__ = ["process_file"]
