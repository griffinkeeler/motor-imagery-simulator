from pathlib import Path

from src.bci.io.loaders import create_raw


_subject_dict = {
    1: "aa",
    2: "al",
    3: "av",
    4: "aw",
    5: "ay",
}


def _save_processed_data(
    subject: int,
    l_freq=8,
    h_freq=30,
):
    """
    Saves data to a .fif file.

    Args:
        mat_file: The name of the mat file.
        subject: Number of the subject.
        l_freq: Low pass frequency.
        h_freq: High pass frequency.

    Returns:
        None
    """
    here = Path(__file__).parents[2]
    # Creates Raw object
    raw = create_raw(f"{here}/data/raw/data_set_IVa_{_subject_dict[subject]}.mat")

    # Filter raw data
    raw.filter(l_freq=l_freq, h_freq=h_freq)

    # Saves data to file
    raw.save(
        f"{here}/data/interim/sub0{subject}/sub0{subject}_preproc_raw.fif",
        overwrite=True,
    )


_save_processed_data(1)
_save_processed_data(2)
_save_processed_data(3)
_save_processed_data(4)
_save_processed_data(5)
