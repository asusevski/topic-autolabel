import os
import tempfile
from pathlib import Path
from typing import List, Optional

import cv2
import numpy as np


def extract_video_frames(
    video_path: str | Path,
    frame_interval: int = 16,
    output_dir: Optional[str | Path] = None,
) -> tuple[str, List[str]]:
    """
    Extract frames from a video file at specified intervals.
    
    Args:
        video_path: Path to the video file
        frame_interval: Extract every nth frame (default: 16)
        output_dir: Directory to save frames (default: creates temporary directory)
    
    Returns:
        tuple[str, List[str]]: Tuple containing (directory path, list of frame paths)
    
    Raises:
        FileNotFoundError: If video file doesn't exist
        ValueError: If video file can't be opened
        RuntimeError: If no frames could be extracted
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Create output directory if not provided
    if output_dir is None:
        output_dir = tempfile.mkdtemp()
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
    # Open video file
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    frame_paths = []
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:  # End of video
                break
                
            if frame_count % frame_interval == 0:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Save frame as PNG
                frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                frame_paths.append(frame_path)
                
            frame_count += 1
    
    finally:
        cap.release()
    
    if not frame_paths:
        raise RuntimeError(f"No frames could be extracted from video: {video_path}")
        
    return str(output_dir), frame_paths


def cleanup_frames(directory: str | Path) -> None:
    """
    Clean up the extracted frames and their directory.
    
    Args:
        directory: Path to the directory containing extracted frames
    """
    directory = Path(directory)
    if directory.exists():
        for file in directory.glob("*.png"):
            file.unlink()
        directory.rmdir()
