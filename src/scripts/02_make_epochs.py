import mne

from pathlib import Path

from src.bci.io.loaders import create_events


def _save_epoch_data(subject: str,
                     session: str):
    """
    Saves the epoch data for a given subject.

    Args:
        subject: The number of the subject.

    Returns:
        None
    """
    here = Path(__file__).parents[2]

    raw = mne.io.read_raw_fif(
        f"{here}/data/interim/{session}/eval_{subject}_preproc_raw.fif"
    )
    events = create_events(
        f"{here}/data/raw/{session}/BCICIV_calib_ds1{subject}_1000Hz.mat"
    )

    event_dict = {"right hand": 0, "left hand": 1}

    epochs = mne.Epochs(
        raw,
        events,
        tmin=-1.5,
        tmax=4.5,
        event_id=event_dict,
        preload=True,
        baseline=None,
    )

    epochs.save(
        f"{here}/data/processed/{session}/{session}_{subject}_epochs-epo.fif",
        overwrite=True,
    )

_save_epoch_data(subject='a', session='calibration')
_save_epoch_data(subject='b', session='calibration')
_save_epoch_data(subject='c', session='calibration')
_save_epoch_data(subject='d', session='calibration')
_save_epoch_data(subject='e', session='calibration')
_save_epoch_data(subject='f', session='calibration')
_save_epoch_data(subject='g', session='calibration')
