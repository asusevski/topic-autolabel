from pathlib import Path
from typing import Optional, Union, Literal, Tuple
import pandas as pd


DataType = Literal['text', 'image', 'video']

def detect_file_type(filepath: Union[str, Path]) -> DataType:
    """Detect the type of file based on extension and content."""
    filepath = Path(filepath)
    suffix = filepath.suffix.lower()
    
    if suffix in {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}:
        return 'image'
    elif suffix in {'.mp4', '.avi', '.mov', '.mkv'}:
        return 'video'
    elif suffix in {'.csv', '.txt'}:
        return 'text'
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def load_data(
    filepath: Union[str, Path], text_column: Optional[str] = None
)  -> Tuple[DataType, pd.DataFrame]:
    """
    Load data from various file types.
    
    Args:
        filepath: Path to the file
        text_column: Column name for text data (CSV files)
        frame_sample_rate: For videos, sample every nth frame
        max_frames: Maximum number of frames to sample from video
        
    Returns:
        - DataFrame for text data
        - PIL Image for image data
        - List of PIL Images for video data
    """
    #TODO: handle directory of images/videos, csvs
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"No file found at {filepath}")

    file_type = detect_file_type(filepath)
    
    if file_type == 'text':
        df = pd.read_csv(filepath)
        if text_column and text_column not in df.columns:
            raise ValueError(f"Column {text_column} not found in the CSV file")
        return "text", df
        
    elif file_type == 'image':
        #TODO: process image for transformers implementations
        return "image", pd.DataFrame({"filepath": [filepath]})
        
    else:
        #TODO: process video for transformers implementations
        return "video", pd.DataFrame({"filepath": [filepath]})
