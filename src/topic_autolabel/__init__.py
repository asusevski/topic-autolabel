from pathlib import Path
from typing import List, Optional, Tuple, TypeGuard, overload

import ollama
import pandas as pd
from huggingface_hub import repo_info
from huggingface_hub.utils import HFValidationError, RepositoryNotFoundError

from .core import TopicLabeler, load_data


def process_file(
    filepath: Optional[str | Path],
    text_column: Optional[str] = None,
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
        num_labels: Number of labels to generate unsupervised
        candidate_labels: List of predefined labels to choose from (optional)

    Returns:
        DataFrame with a new 'label' column containing the generated labels
    """
    if filepath is None and df is None or filepath is not None and df is not None:
        raise ValueError("One of df or filepath must be provided")

    if (
        num_labels is None
        and candidate_labels is None
        or num_labels is not None
        and candidate_labels is not None
    ):
        raise ValueError("One of num_labels or candidate_labels must be provided")

    if filepath is not None:
        dtype, df = load_data(filepath, text_column)
    else:
        dtype = "text"

    # Find out if model points to an ollama model running on a server or a huggingface model
    try:
        repo_info(model_name)
        huggingface_model = model_name
        ollama_model = ""
        # TODO: make this not dumb
        if dtype != "text":
            raise ValueError(
                "Found non-text data and a HuggingFace model path, please only run with Ollama."
            )
    except (RepositoryNotFoundError, HFValidationError):
        # check for ollama
        valid_models = [str(x.model) for x in ollama.list().models]
        if model_name not in valid_models:
            raise ValueError(
                f"Model '{model_name}' not found in the HuggingFace Hub nor is it currently being served by ollama. Models found: {valid_models}"
            )
        else:
            try:
                ollama.chat(model_name)
            except ConnectionError:
                raise ValueError(
                    f"Ollama model {model_name} detected, but server unavailable. Ensure server is available. Models found: {valid_models}"
                )
        huggingface_model = ""
        ollama_model = model_name

    # Initialize the labeler
    labeler = TopicLabeler(
        huggingface_model=huggingface_model,
        ollama_model=ollama_model,
        batch_size=batch_size,
        dtype=dtype,
    )

    # Generate labels
    if dtype == "text":
        labels = labeler.generate_labels(
            texts=df[text_column].tolist(),
            num_labels=num_labels,
            candidate_labels=candidate_labels,
        )
        # TODO: make sure this column doesn't already exist in df
        df["label"] = labels
    else:
        labels = labeler.generate_labels(
            df=df,
            num_labels=num_labels,
            candidate_labels=candidate_labels,
        )
        # TODO: make sure this column doesn't already exist in df
        df["label"] = labels
    # Add labels to dataframe

    return df


__all__ = ["process_file"]
