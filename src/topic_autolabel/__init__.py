from typing import List, Optional
import pandas as pd
from .core.data_loader import load_data
from .core.labeler import TopicLabeler
from huggingface_hub import repo_info
from huggingface_hub.utils import RepositoryNotFoundError
import ollama


def process_file(
    filepath: Optional[str],
    text_column: str,
    df: Optional[pd.DataFrame] = None,
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
    num_labels: Optional[int] = 5,
    candidate_labels: Optional[List[str]] = None,
    batch_size: Optional[int] = 8,
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
    try:
        assert filepath is not None or df is not None
    except AssertionError:
        raise ValueError("One of filepath or df must be passed to the function.")

    # Load the data
    if df is None:
        df = load_data(filepath, text_column)

    # Check if the text column exists
    try:
        assert text_column in df.columns
    except AssertionError:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")
    
    # Find out if model points to an ollama model running on a server or a huggingface model
    try:
        repo_info(model_name)
        huggingface_model = model_name
        ollama_model = ""
    except RepositoryNotFoundError:
        # check for ollama 
        valid_models = [str(x.model.split(':')[0]) for x in ollama.list().models]
        if model_name not in valid_models:
            raise ValueError(f"Model '{model_name}' not found in the HuggingFace Hub nor is it currently being served by ollama.")
        else:
            try:
                ollama.chat(model_name)
            except:
                raise ValueError(f"Ollama model {model_name} detected, but server unavailable. Ensure server is available.")
        huggingface_model = "" 
        ollama_model = model_name

    # Initialize the labeler
    labeler = TopicLabeler(
        huggingface_model=huggingface_model,
        ollama_model=ollama_model,
        batch_size=batch_size
    )

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
