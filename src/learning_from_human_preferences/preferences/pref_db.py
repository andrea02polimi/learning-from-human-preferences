"""
Preference database utilities.

Refactored version of the original implementation with:
- modern Python style
- TensorBoard logging
- same logical behaviour
"""

import copy
import gzip
import pickle
import queue
import time
import zlib
from dataclasses import dataclass, field
from threading import Lock, Thread
from typing import Dict, List, Tuple, Iterator

import numpy as np
from torch.utils.tensorboard import SummaryWriter


# ----------------------------------------------------------
# Segment
# ----------------------------------------------------------

@dataclass
class Segment:
    """
    A short recording of agent behaviour consisting of frames and rewards.
    """

    frames: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    hash: int | None = None

    def append(self, frame: np.ndarray, reward: float) -> None:
        self.frames.append(frame)
        self.rewards.append(reward)

    def finalize(self, seg_id: int | None = None) -> None:
        if seg_id is not None:
            self.hash = seg_id
        else:
            self.hash = hash(np.asarray(self.frames).tobytes())

    def __len__(self) -> int:
        return len(self.frames)


# ----------------------------------------------------------
# Compressed dictionary
# ----------------------------------------------------------

class CompressedDict:
    """
    Dictionary storing compressed objects.
    """

    def __init__(self):
        self._store: Dict[int, bytes] = {}

    def __getitem__(self, key: int):
        return pickle.loads(zlib.decompress(self._store[key]))

    def __setitem__(self, key: int, value):
        self._store[key] = zlib.compress(pickle.dumps(value))

    def __delitem__(self, key: int):
        del self._store[key]

    def __iter__(self) -> Iterator[int]:
        return iter(self._store)

    def __len__(self) -> int:
        return len(self._store)

    def keys(self):
        return self._store.keys()


# ----------------------------------------------------------
# Preference database
# ----------------------------------------------------------

class PrefDB:
    """
    Circular database storing preferences over pairs of segments.
    """

    def __init__(self, maxlen: int):

        self.segments: CompressedDict = CompressedDict()
        self.segment_references: Dict[int, int] = {}

        self.preferences: List[Tuple[int, int, np.ndarray]] = []

        self.maxlen = maxlen

    # ------------------------------------------------------

    def append(self, seg1, seg2, preference):

        key1 = hash(np.asarray(seg1).tobytes())
        key2 = hash(np.asarray(seg2).tobytes())

        for key, segment in zip([key1, key2], [seg1, seg2]):

            if key not in self.segments.keys():
                self.segments[key] = segment
                self.segment_references[key] = 1
            else:
                self.segment_references[key] += 1

        self.preferences.append((key1, key2, preference))

        if len(self.preferences) > self.maxlen:
            self._delete_first()

    # ------------------------------------------------------

    def _delete_first(self):
        self.delete_preference(0)

    # ------------------------------------------------------

    def delete_preference(self, index: int):

        if index >= len(self.preferences):
            raise IndexError(f"Preference {index} does not exist")

        key1, key2, _ = self.preferences[index]

        for key in [key1, key2]:

            if self.segment_references[key] == 1:
                del self.segments[key]
                del self.segment_references[key]
            else:
                self.segment_references[key] -= 1

        del self.preferences[index]

    # ------------------------------------------------------

    def __len__(self) -> int:
        return len(self.preferences)

    # ------------------------------------------------------

    def save(self, path: str):

        with gzip.open(path, "wb") as f:
            pickle.dump(self, f)

    # ------------------------------------------------------

    @staticmethod
    def load(path: str):

        with gzip.open(path, "rb") as f:
            return pickle.load(f)


# ----------------------------------------------------------
# Preference buffer
# ----------------------------------------------------------

class PrefBuffer:
    """
    Handles asynchronous reception of preferences.
    """

    def __init__(self, db_train: PrefDB, db_val: PrefDB, log_dir: str | None = None):

        self.train_db = db_train
        self.val_db = db_val

        self.lock = Lock()

        self._stop_flag = False
        self._thread: Thread | None = None

        self.step = 0

        self.writer = SummaryWriter(log_dir) if log_dir is not None else None

    # ------------------------------------------------------

    def start_recv_thread(self, pref_queue):

        self._stop_flag = False

        self._thread = Thread(
            target=self._recv_preferences,
            args=(pref_queue,),
            daemon=True
        )

        self._thread.start()

    # ------------------------------------------------------

    def stop_recv_thread(self):

        self._stop_flag = True

        if self._thread is not None:
            self._thread.join()

    # ------------------------------------------------------

    def _recv_preferences(self, pref_queue):

        received = 0

        while not self._stop_flag:

            try:
                seg1, seg2, preference = pref_queue.get(timeout=1)
            except queue.Empty:
                continue

            received += 1
            self.step += 1

            validation_fraction = (
                self.val_db.maxlen /
                (self.val_db.maxlen + self.train_db.maxlen)
            )

            with self.lock:

                if np.random.rand() < validation_fraction:
                    self.val_db.append(seg1, seg2, preference)
                else:
                    self.train_db.append(seg1, seg2, preference)

                # ----------------------------------
                # TensorBoard logging
                # ----------------------------------

                if self.writer is not None:
                    self.writer.add_scalar(
                        "preferences/train_db_size", len(self.train_db), self.step,
                    )
                    self.writer.add_scalar(
                        "preferences/val_db_size", len(self.val_db), self.step,
                    )
                    self.writer.add_scalar(
                        "preferences/total_received", received, self.step,
                    )

    # ------------------------------------------------------

    def train_db_len(self) -> int:
        return len(self.train_db)

    def val_db_len(self) -> int:
        return len(self.val_db)

    # ------------------------------------------------------

    def get_dbs(self):

        with self.lock:

            train_copy = copy.deepcopy(self.train_db)
            val_copy = copy.deepcopy(self.val_db)

        return train_copy, val_copy

    # ------------------------------------------------------

    def wait_until_len(self, minimum_length: int):

        while True:

            with self.lock:
                train_len = len(self.train_db)
                val_len = len(self.val_db)

            if train_len >= minimum_length and val_len > 0:
                break

            print(f"Waiting for preferences; {train_len} collected")

            time.sleep(5.0)