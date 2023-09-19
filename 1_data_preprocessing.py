import pandas as pd
import math
import cv2
from pathlib import Path
import h5py
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.animation import FuncAnimation
from glob import glob
def load_h5_matrix(fp: Path):
    """Load a matrix from an h5 file."""
    with h5py.File(fp, 'r') as f:
        occupancy_matrix = f['track_occupancy'][:] # binary matrix (frame#, animal#)
        tracks_matrix = f['tracks'][0].T # (frame#, node#, coord)
        node_names = [n.decode() for n in f["node_names"][:]]
    return node_names, occupancy_matrix, tracks_matrix

def inter_2d(data: np.ndarray) -> np.ndarray:
    """Interpolate a 2D numpy array using linear interpolation.
    Args:
    - data (np.ndarray): A 2D numpy array with 2 columns (x and y).
    Returns:
    - np.ndarray: The interpolated 2D numpy array.
    """
    data_inter = np.empty_like(data)
    non_nan_idx = np.where(~np.isnan(data[:, 0]))[0]
    non_nan_values = data[non_nan_idx, :]
    # Interpolate the values at all indices
    for dim in range(2):
        data_inter[:, dim] = np.interp(range(data.shape[0]), non_nan_idx, non_nan_values[:, dim])
    return data_inter


def tracks_to_chunks(df:pd.DataFrame, data: np.ndarray):
    chunks = []
    for _, row in df.iterrows():
        start_frame, end_frame = row['FrameStart'], row['FrameStop']
        chunk = data[start_frame:end_frame+1]  # slice the data array using start and end frames
        chunks.append(chunk)
    return chunks

def plot_chunk(chunk: np.ndarray, save_path: Path=None):
    ''' 
    Plot trail from h5 file
    args:
        chunk: np.ndarray of shape (chunk_size, 5, 2)
    '''

    colors = ['red', 'green', 'blue', 'purple', 'orange']
    plt.figure(figsize=(8, 6))
    for i in range(chunk.shape[1]):
        x = chunk[:, i, 0]
        y = chunk[:, i, 1]
        plt.plot(x, y, color=colors[i])
    plt.xlim(0, 550)
    plt.ylim(0, 500)
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_chunk_anim(chunk: np.ndarray, save_path: Path=None):
    ''' 
    Plot trail from h5 file
    args:
        chunk: np.ndarray of shape (n, 5, 2)
    '''
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 550)
    ax.set_ylim(0, 500)
    colors = ['red', 'green', 'blue', 'purple', 'orange']
    lines = [ax.plot([], [], color=color)[0] for color in colors]
    anim_speed = 5
    def init():
        for line in lines:
            line.set_data([], [])
        return lines

    def update(frame):
        frame *= anim_speed # speed up the animation
        for i, line in enumerate(lines):
            x = chunk[:frame, i, 0]
            y = chunk[:frame, i, 1]
            line.set_data(x, y)
        return lines

    anim = FuncAnimation(fig, update, frames=chunk.shape[0] // anim_speed, init_func=init, blit=True, interval=10)

    if save_path:
        anim.save(save_path, writer='imagemagick')

    plt.show()

def write_chunks_to_h5(chunks, fp):
    # Save chunks to a single HDF5 file
    with h5py.File(fp, 'w') as f:
        for idx, chunk in enumerate(chunks):
            f.create_dataset(str(idx), data=chunk)

def read_chunks_from_h5(fp: Path):
    chunks = []
    with h5py.File(fp, 'r') as f:
        sorted_keys = sorted(f.keys(), key=int)
        for key in sorted_keys:
            chunk = f[key][:]
            chunks.append(chunk)
    return chunks


def process_video(h5_path, csv_path, output_path):
    df = pd.read_csv(csv_path)
    node_names, occupancy_matrix, tracks_matrix = load_h5_matrix(h5_path)

    # Interpolate tracks_matrix on axis=1
    inter_tracks = np.empty_like(tracks_matrix)
    total_chunks = tracks_matrix.shape[1]
    for i in range(total_chunks):
        inter_tracks[:, i, :] = inter_2d(tracks_matrix[:, i, :])

    # Cut tracks_matrix into chunks
    chunks = tracks_to_chunks(df, inter_tracks)
    print(len(chunks))
    # Plot chunks and save outputs
    # rand_sample = np.random.randint(0, total_chunks, size=1)
    # save_track_path = output_path / 'track_tracing'
    # save_track_path.mkdir(parents=True, exist_ok=True)
    # for idx in rand_sample:
    #     plot_chunk_anim(chunks[idx], save_track_path / f'track_{idx}.gif')
    #     plot_chunk(chunks[idx], save_track_path / f'track_{idx}.png')

    # Save chunks to h5
    save_h5_path = output_path / 'saved_chunks.h5'
    write_chunks_to_h5(chunks, save_h5_path)


data_path = Path('./data')

# Loop through each video folder
video_folders = [folder for folder in data_path.iterdir() if folder.is_dir()]
print(len(video_folders))
for video_folder in video_folders:
     # Grab the first (and only) h5 file in the folder
    print(video_folder)
    h5_path = next(video_folder.glob("*.h5"), None)

    # Grab the first (and only) csv file in the folder
    csv_path = next(video_folder.glob("*.csv"), None)
    
    output_path = Path('./output') / video_folder.name
    output_path.mkdir(parents=True, exist_ok=True)

    process_video(h5_path, csv_path, output_path)