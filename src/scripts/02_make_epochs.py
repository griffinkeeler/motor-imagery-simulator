import mne

from pathlib import Path

from src.bci.io.loaders import create_events


# events→epochs (tmin/tmax)→save X,y/Epochs
_subject_dict = {
    1: "aa",
    2: "al",
    3: "av",
    4: "aw",
    5: "ay",
}


def _save_epoch_data(subject: int):
    """
    Saves the epoch data for a given subject.

    Args:
        subject: The number of the subject.

    Returns:
        None
    """
    here = Path(__file__).parents[2]

    raw = mne.io.read_raw_fif(
        f"{here}/data/interim/sub0{subject}" f"/sub0{subject}_preproc_raw.fif"
    )
    events = create_events(
        f"{here}/data/raw/data_set_IVa_{_subject_dict[subject]}.mat"
    )

    event_dict = {"right hand": 1, "foot": 2}

    epochs = mne.Epochs(
        raw,
        events,
        tmin=0.0,
        tmax=4.0,
        event_id=event_dict,
        preload=True,
        baseline=None,
    )

    epochs.save(
        f"{here}/data/processed/sub0{subject}/sub0" f"{subject}_epochs-epo.fif",
        overwrite=True,
    )


_save_epoch_data(1)
_save_epoch_data(2)
_save_epoch_data(3)
_save_epoch_data(4)
_save_epoch_data(5)
