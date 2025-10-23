import mne.channels

from pathlib import Path

from src.bci.io.loaders import create_raw


def _save_processed_data(
    subject: str,
    session: str,
    l_freq=8,
    h_freq=30,
):
    """
    Saves data to a .fif file.

    Args:
        subject: Number of the subject.
        l_freq: Low pass frequency.
        h_freq: High pass frequency.

    Returns:
        None
    """
    here = Path(__file__).parents[2]
    # Creates Raw object
    raw = create_raw(file_path=f"{here}/data/raw/{session}/BCICIV_calib_ds1{subject}_1000Hz.mat")

    # Filter raw data
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Saves data to file
    raw.save(
        f"{here}/data/interim/{session}/eval_{subject}_preproc_raw.fif",
        overwrite=True,
    )

_save_processed_data(subject='a', session='calibration')
_save_processed_data(subject='b', session='calibration')
_save_processed_data(subject='c', session='calibration')
_save_processed_data(subject='d', session='calibration')
_save_processed_data(subject='e', session='calibration')
_save_processed_data(subject='f', session='calibration')
_save_processed_data(subject='g', session='calibration')



