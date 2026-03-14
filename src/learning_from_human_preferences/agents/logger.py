import os
import sys
import json
import shutil
import tempfile
import datetime
import os.path as osp
import numpy as np

from torch.utils.tensorboard import SummaryWriter


LOG_OUTPUT_FORMATS = ["stdout", "log", "json"]

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40
DISABLED = 50


# =========================================================
# Output formats
# =========================================================

class OutputFormat:

    def writekvs(self, kvs):
        raise NotImplementedError

    def writeseq(self, args):
        pass

    def close(self):
        pass


# =========================================================
# Human readable output
# =========================================================

class HumanOutputFormat(OutputFormat):

    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):

        key2str = {}

        for key, val in sorted(kvs.items()):

            if isinstance(val, float):
                valstr = "%-8.3g" % val
            else:
                valstr = str(val)

            key2str[self._truncate(key)] = self._truncate(valstr)

        keywidth = max(map(len, key2str.keys()))
        valwidth = max(map(len, key2str.values()))

        dashes = "-" * (keywidth + valwidth + 7)

        lines = [dashes]

        for key, val in sorted(key2str.items()):

            lines.append(
                "| %s%s | %s%s |" %
                (key,
                 " " * (keywidth - len(key)),
                 val,
                 " " * (valwidth - len(val)))
            )

        lines.append(dashes)

        self.file.write("\n".join(lines) + "\n")

        self.file.flush()

    def writeseq(self, args):

        for arg in args:
            self.file.write(str(arg))

        self.file.write("\n")

        self.file.flush()

    def _truncate(self, s):

        return s[:20] + "..." if len(s) > 23 else s

    def close(self):

        if self.file not in (sys.stdout, sys.stderr):
            self.file.close()


# =========================================================
# JSON output
# =========================================================

class JSONOutputFormat(OutputFormat):

    def __init__(self, file):
        self.file = file

    def writekvs(self, kvs):

        kvs = dict(kvs)

        for k, v in sorted(kvs.items()):

            if hasattr(v, "dtype"):
                v = v.tolist()
                kvs[k] = float(v)

        self.file.write(json.dumps(kvs) + "\n")

        self.file.flush()

    def close(self):
        self.file.close()


# =========================================================
# TensorBoard output
# =========================================================

class TensorBoardOutputFormat(OutputFormat):

    def __init__(self, logdir):

        os.makedirs(logdir, exist_ok=True)

        self.writer = SummaryWriter(logdir)

        self.step = 1

    def writekvs(self, kvs):

        for k, v in kvs.items():

            try:
                self.writer.add_scalar(k, float(v), self.step)
            except Exception:
                pass

        self.writer.flush()

        self.step += 1

    def close(self):
        self.writer.close()


# =========================================================
# Factory
# =========================================================

def make_output_format(fmt, logdir):

    os.makedirs(logdir, exist_ok=True)

    if fmt == "stdout":
        return HumanOutputFormat(sys.stdout)

    elif fmt == "log":
        return HumanOutputFormat(open(osp.join(logdir, "log.txt"), "wt"))

    elif fmt == "json":
        return JSONOutputFormat(open(osp.join(logdir, "progress.json"), "wt"))

    elif fmt == "tensorboard":
        return TensorBoardOutputFormat(osp.join(logdir, "tb"))

    else:
        raise ValueError("Unknown format specified: %s" % fmt)


# =========================================================
# Logger
# =========================================================

class Logger:

    DEFAULT = None
    CURRENT = None

    def __init__(self, dir, output_formats):

        self.name2val = {}
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats

    def logkv(self, key, val):

        self.name2val[key] = val

    def dumpkvs(self):

        if self.level == DISABLED:
            return

        for fmt in self.output_formats:
            fmt.writekvs(self.name2val)

        self.name2val.clear()

    def log(self, *args, level=INFO):

        if self.level <= level:

            for fmt in self.output_formats:
                fmt.writeseq(args)

    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):

        for fmt in self.output_formats:
            fmt.close()


Logger.DEFAULT = Logger(
    dir=None,
    output_formats=[HumanOutputFormat(sys.stdout)],
)

Logger.CURRENT = Logger.DEFAULT


# =========================================================
# Public API
# =========================================================

def logkv(key, val):
    Logger.CURRENT.logkv(key, val)


def logkvs(d):
    for k, v in d.items():
        logkv(k, v)


def dumpkvs():
    Logger.CURRENT.dumpkvs()


def getkvs():
    return Logger.CURRENT.name2val


def log(*args, level=INFO):
    Logger.CURRENT.log(*args, level=level)


def debug(*args):
    log(*args, level=DEBUG)


def info(*args):
    log(*args, level=INFO)


def warn(*args):
    log(*args, level=WARN)


def error(*args):
    log(*args, level=ERROR)


def set_level(level):
    Logger.CURRENT.set_level(level)


def get_dir():
    return Logger.CURRENT.get_dir()


record_tabular = logkv
dump_tabular = dumpkvs


# =========================================================
# configure
# =========================================================

def configure(dir=None, format_strs=None):

    assert Logger.CURRENT is Logger.DEFAULT

    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")

    if dir is None:

        dir = osp.join(
            tempfile.gettempdir(),
            datetime.datetime.now().strftime(
                "openai-%Y-%m-%d-%H-%M-%S-%f"
            ),
        )

    if format_strs is None:
        format_strs = LOG_OUTPUT_FORMATS

    output_formats = [make_output_format(f, dir) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats)

    log("Logging to %s" % dir)


# auto configure (original behavior)

if os.getenv("OPENAI_LOGDIR"):
    configure(dir=os.getenv("OPENAI_LOGDIR"))


def reset():

    Logger.CURRENT = Logger.DEFAULT

    log("Reset logger")


# =========================================================
# session context manager (original API)
# =========================================================

class session:

    def __init__(self, dir=None, format_strs=None):

        self.dir = dir
        self.format_strs = format_strs
        self.prev_logger = None

    def __enter__(self):

        self.prev_logger = Logger.CURRENT

        configure(self.dir, self.format_strs)

    def __exit__(self, *args):

        Logger.CURRENT.close()

        Logger.CURRENT = self.prev_logger