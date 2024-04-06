import os
import numpy as np
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor


def windowed_average(data, window_size, padding_type="reflect"):
    """
    Compute windowed average with flexible padding
    ---
    Args:
    - data: Data to be processed
    - window_size: Window size of the averaging kernel
    - padding_type: Type of padding to be applied

    Returns:
    - data_averaged: Data with windowed average applied
    """

    # Add reflect padding to data
    data_padded = np.pad(data, (window_size // 2, window_size // 2 - 1), padding_type)

    # Convolve with averaging kernel
    kernel = np.ones(window_size) / window_size
    data_averaged = np.convolve(data_padded, kernel, mode="valid")

    return data_averaged


def round_to(number, to=5, up=True):
    """
    Rounds number to nearest value, either up or down.
    ---
    Args:
    - number: Number to be rounded
    - to: Value to round to
    - up: Whether to round up or down

    Returns:
    - rounded: Rounded number
    """
    rounded = np.ceil(number / to) if up else np.floor(number / to)
    return rounded * to


def plot_step_rewards(step_rewards, kernel_size=500, save_path=None):
    """
    Plots step rewards with smoothed curve.
    ---
    Args:
    - step_rewards: List of step rewards
    - kernel_size: Size of the averaging kernel
    - save_path: Path to save the plot

    Returns:
    - fig: Figure object
    - ax: Axes object
    """

    # Plot step rewards
    smooth_step_rewards = windowed_average(step_rewards, kernel_size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # ax.plot(step_rewards, label="raw")
    ax.plot(smooth_step_rewards, label="smoothed")
    ax.set_ylim(
        min(0, np.min(smooth_step_rewards)),
        round_to(np.max(smooth_step_rewards), 5),
    )
    ax.grid(True)
    ax.set_xlabel("Move")
    ax.set_ylabel(f"Mean Reward\n({kernel_size} steps)")
    # ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, facecolor="white")
    return fig, ax


def plot_episode_rewards(episode_rewards, kernel_size=25, save_path=None):
    """
    Plots episode rewards with smoothed curve.
    ---
    Args:
    - episode_rewards: List of episode rewards
    - kernel_size: Size of the averaging kernel
    - save_path: Path to save the plot

    Returns:
    - fig: Figure object
    - ax: Axes object
    """

    smooth_episode_rewards = windowed_average(episode_rewards, kernel_size)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    # ax.plot(episode_rewards, label="raw")
    ax.plot(smooth_episode_rewards, label="smoothed")
    ax.set_ylim(
        min(0, np.min(smooth_episode_rewards)),
        round_to(np.max(smooth_episode_rewards), 10),
    )
    ax.grid(True)
    ax.set_xlabel("Episode")
    ax.set_ylabel(f"Mean Reward\n({kernel_size} episodes)")
    # ax.legend()
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path, dpi=300, facecolor="white")
    return fig, ax


def make_safe_filename(filename):
    # Define a regular expression to remove characters not allowed in filenames
    safe_chars = re.compile(r'[^a-zA-Z0-9_.-]')
    # Replace disallowed characters with underscores
    safe_filename = re.sub(safe_chars, '_', filename)
    return safe_filename