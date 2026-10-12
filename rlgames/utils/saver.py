import os
import pickle

import jax
import numpy as np


def save_pickle(path: str, obj) -> None:
    """Pickle `obj` with jax arrays converted to numpy. Writes to a temporary file and
    renames it, so readers (e.g. the play server) never see a partial file."""
    obj = jax.tree.map(lambda x: np.asarray(x) if isinstance(x, jax.Array) else x, obj)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)
