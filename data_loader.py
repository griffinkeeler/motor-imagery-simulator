import mne
import numpy as np

from scipy.io import loadmat


def create_raw(file_path: str):
    """
    Creates an MNE raw object from a .mat file containing EEG data and
    related information.

    Parameters
    ----------

    file_path : str
        The path to the .mat file.

    Returns
    -------
    raw : mne.RawArray
        The MNE raw object.
    """

    file = loadmat(file_path)

    # Continuous EEG Signal, shape (118, 298458)
    data = file["cnt"].T

    # Shape: (1, 1)
    nfo = file["nfo"]

    # Shape: (1, 118)
    ch_lab_array = nfo["clab"][0, 0]

    # List of channels labels
    ch_lab = [ch.item() for ch in ch_lab_array[0, :]]

    # Sampling frequency: 1000 Hz
    fs = float(nfo["fs"][0, 0].item())

    # Create MNE's info object
    info = mne.create_info(
        ch_names=ch_lab,
        sfreq=fs,
        ch_types="eeg",
    )

    # Create MNE's raw object
    return mne.io.RawArray(
        data=data,
        info=info,
    )


def create_events(file_path: str):
    """
    Creates an events array .mat file containing EEG data and
    related information.

    Parameters
    ----------
    file_path : str
        The path to the .mat file.


    Returns
    -------
    events : np.array
        An array of markers of trial onsets and labels (shape: n_events, 3).
    """

    # Events shape: (n_events, 3)
    _file = loadmat(file_path)

    # Shape: (1, 1)
    mrk = _file["mrk"]

    # Shape: (1, 280)
    trials = mrk["pos"][0, 0].squeeze().astype(int)

    # Shape: (280,)
    targets = mrk["y"][0, 0].squeeze()

    # Shape: (168,)
    y = targets[~np.isnan(targets)].astype(int)

    # Shape: (168,)
    clean_trials = trials[y]

    # Shape: (168,)
    zeros_arr = np.zeros_like(y)

    # Shape: (168, 3)
    return np.column_stack([clean_trials, zeros_arr, y])
