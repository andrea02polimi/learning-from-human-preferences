"""
CLI-based interface for collecting human preferences between trajectory segments.
Refactored version preserving the original logic.
"""

import logging
import queue
import time
from itertools import combinations
from multiprocessing import Queue
from random import shuffle
from typing import List, Optional, Tuple

import numpy as np
from torch.utils.tensorboard import SummaryWriter

from learning_from_human_preferences.envs.utils import VideoRenderer


# ----------------------------------------------------------
# TensorBoard writer
# ----------------------------------------------------------

writer = SummaryWriter("runs/preferences_interface")


# ----------------------------------------------------------
# Preference Interface
# ----------------------------------------------------------

class PrefInterface:
    """
    CLI interface that queries the user for preferences between trajectory segments.
    """

    def __init__(self, synthetic_prefs: bool, max_segs: int, log_dir: str):

        self.video_queue = Queue()

        if not synthetic_prefs:
            self.renderer = VideoRenderer(
                vid_queue=self.video_queue,
                mode=VideoRenderer.restart_on_get_mode,
                zoom=4
            )
        else:
            self.renderer = None

        self.synthetic_prefs = synthetic_prefs

        # Circular buffer state
        self.segment_index = 0
        self.segments: List = []

        # Track tested segment pairs
        self.tested_pairs = set()

        self.max_segments = max_segs

        self.step = 0

    # ------------------------------------------------------

    def stop_renderer(self):

        if self.renderer:
            self.renderer.stop()

    # ------------------------------------------------------

    def run(self, segment_pipe, preference_pipe):

        while len(self.segments) < 2:
            print("Preference interface waiting for segments")
            time.sleep(5.0)
            self.receive_segments(segment_pipe)

        while True:

            segment_pair = None

            while segment_pair is None:

                try:
                    segment_pair = self.sample_segment_pair()

                except IndexError:

                    print("Preference interface ran out of untested segments; waiting...")
                    time.sleep(1.0)
                    self.receive_segments(segment_pipe)

            seg1, seg2 = segment_pair

            logging.debug(
                "Querying preference for segments %s and %s",
                seg1.hash,
                seg2.hash,
            )

            if not self.synthetic_prefs:

                preference = self.ask_user(seg1, seg2)

            else:

                reward1 = sum(seg1.rewards)
                reward2 = sum(seg2.rewards)

                if reward1 > reward2:
                    preference = (1.0, 0.0)
                elif reward1 < reward2:
                    preference = (0.0, 1.0)
                else:
                    preference = (0.5, 0.5)

            if preference is not None:

                preference_pipe.put(
                    (seg1.frames, seg2.frames, preference)
                )

            self.receive_segments(segment_pipe)

    # ------------------------------------------------------

    def receive_segments(self, segment_pipe):

        """
        Receive segments from the queue into a circular buffer.
        """

        max_wait_seconds = 0.5

        start_time = time.time()
        received = 0

        while time.time() - start_time < max_wait_seconds:

            try:
                segment = segment_pipe.get(timeout=max_wait_seconds)

            except queue.Empty:
                break

            if len(self.segments) < self.max_segments:

                self.segments.append(segment)

            else:

                self.segments[self.segment_index] = segment

                self.segment_index = (
                    (self.segment_index + 1) % self.max_segments
                )

            received += 1

        # TensorBoard logging
        self.step += 1

        writer.add_scalar("segments/index", self.segment_index, self.step)
        writer.add_scalar("segments/received", received, self.step)
        writer.add_scalar("segments/buffer_size", len(self.segments), self.step)

    # ------------------------------------------------------

    def sample_segment_pair(self):

        """
        Sample a random pair of segments that has not been tested yet.
        """

        indices = list(range(len(self.segments)))

        shuffle(indices)

        possible_pairs = combinations(indices, 2)

        for i1, i2 in possible_pairs:

            seg1 = self.segments[i1]
            seg2 = self.segments[i2]

            if (
                (seg1.hash, seg2.hash) not in self.tested_pairs
                and (seg2.hash, seg1.hash) not in self.tested_pairs
            ):

                self.tested_pairs.add((seg1.hash, seg2.hash))
                self.tested_pairs.add((seg2.hash, seg1.hash))

                return seg1, seg2

        raise IndexError("No segment pairs yet untested")

    # ------------------------------------------------------

    def ask_user(self, seg1, seg2) -> Optional[Tuple[float, float]]:

        """
        Display segments side-by-side and ask the user for a preference.
        """

        video_frames = []

        segment_length = len(seg1)

        for t in range(segment_length):

            border = np.zeros((84, 10), dtype=np.uint8)

            frame = np.hstack(
                (
                    seg1.frames[t][:, :, -1],
                    border,
                    seg2.frames[t][:, :, -1],
                )
            )

            video_frames.append(frame)

        pause_frames = 7

        for _ in range(pause_frames):
            video_frames.append(np.copy(video_frames[-1]))

        self.video_queue.put(video_frames)

        while True:

            print(f"Segments {seg1.hash} and {seg2.hash}: ")

            choice = input()

            if choice in {"L", "R", "E", ""}:
                break

            print(f"Invalid choice '{choice}'")

        if choice == "L":
            preference = (1.0, 0.0)

        elif choice == "R":
            preference = (0.0, 1.0)

        elif choice == "E":
            preference = (0.5, 0.5)

        else:
            preference = None

        # Clear video display
        self.video_queue.put(
            [np.zeros(video_frames[0].shape, dtype=np.uint8)]
        )

        return preference