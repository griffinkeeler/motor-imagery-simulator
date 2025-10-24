from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.pipeline import make_pipeline

from mne.decoding import CSP


def train_csp_lda(X, y, cfg):
    """Train a CSP+LDA pipeline using fixed hyperparameters."""
    csp = CSP(**cfg["csp"])
    lda = LDA(**cfg["lda"])

    clf = make_pipeline(csp, lda)

    clf.fit(X, y)

    return clf

