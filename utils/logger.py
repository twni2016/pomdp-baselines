import os
import sys
import os.path as osp
import json
import time
import datetime
import dateutil.tz
import tempfile
from collections import OrderedDict, Set

try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

LOG_OUTPUT_FORMATS = ["stdout", "log", "csv", "tensorboard"]
# Also valid: json, tensorboard

DEBUG = 10
INFO = 20
WARN = 30
ERROR = 40

DISABLED = 50


class OrderedSet(Set):
    # https://stackoverflow.com/a/10006674/9072850
    def __init__(self, iterable=()):
        self.d = OrderedDict.fromkeys(iterable)

    def __len__(self):
        return len(self.d)

    def __contains__(self, element):
        return element in self.d

    def __iter__(self):
        return iter(self.d)


class KVWriter(object):
    def writekvs(self, kvs):
        raise NotImplementedError


class SeqWriter(object):
    def writeseq(self, seq):
        raise NotImplementedError


def put_in_middle(str1, str2):
    # Put str1 in str2
    n = len(str1)
    m = len(str2)
    if n <= m:
        return str2
    else:
        start = (n - m) // 2
        return str1[:start] + str2 + str1[start + m :]


class HumanOutputFormat(KVWriter, SeqWriter):
    def __init__(self, filename_or_file):
        if isinstance(filename_or_file, str):
            self.file = open(filename_or_file, "wt")
            self.own_file = True
        else:
            assert hasattr(filename_or_file, "read"), (
                "expected file or str, got %s" % filename_or_file
            )
            self.file = filename_or_file
            self.own_file = False

    def writekvs(self, kvs):
        # Create strings for printing
        key2str = {}
        for (key, val) in sorted(kvs.items()):
            if isinstance(val, float):
                valstr = "%-8.3g" % (val,)
            else:
                valstr = str(val)
            key2str[self._truncate(key)] = self._truncate(valstr)

        # Find max widths
        if len(key2str) == 0:
            print("WARNING: tried to write empty key-value dict")
            return
        else:
            keywidth = max(map(len, key2str.keys()))
            valwidth = max(map(len, key2str.values()))

        # Write out the data
        now = datetime.datetime.now(dateutil.tz.tzlocal())
        timestamp = now.strftime("%Y-%m-%d %H:%M:%S.%f %Z")

        dashes = "-" * (keywidth + valwidth + 7)
        dashes_time = put_in_middle(dashes, timestamp)
        lines = [dashes_time]
        for (key, val) in sorted(key2str.items()):
            lines.append(
                "| %s%s | %s%s |"
                % (
                    key,
                    " " * (keywidth - len(key)),
                    val,
                    " " * (valwidth - len(val)),
                )
            )
        lines.append(dashes)
        self.file.write("\n".join(lines) + "\n")

        # Flush the output to the file
        self.file.flush()

    def _truncate(self, s):
        return s[:30] + "..." if len(s) > 33 else s

    def writeseq(self, seq):
        for arg in seq:
            self.file.write(arg + " ")
        self.file.write("\n")
        self.file.flush()

    def close(self):
        if self.own_file:
            self.file.close()


class JSONOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "wt")

    def writekvs(self, kvs):
        for k, v in sorted(kvs.items()):
            if hasattr(v, "dtype"):
                v = v.tolist()
                kvs[k] = float(v)
        self.file.write(json.dumps(kvs) + "\n")
        self.file.flush()

    def close(self):
        self.file.close()


class CSVOutputFormat(KVWriter):
    def __init__(self, filename):
        self.file = open(filename, "w+t")
        self.keys = []
        self.sep = ","

    def writekvs(self, kvs):
        # Add our current row to the history
        extra_keys = list(OrderedSet(kvs.keys()) - OrderedSet(self.keys))
        if extra_keys:
            self.keys.extend(extra_keys)
            self.file.seek(0)
            lines = self.file.readlines()
            self.file.seek(0)
            for (i, k) in enumerate(self.keys):
                if i > 0:
                    self.file.write(",")
                self.file.write(k)
            self.file.write("\n")
            for line in lines[1:]:
                self.file.write(line[:-1])
                self.file.write(self.sep * len(extra_keys))
                self.file.write("\n")
        for (i, k) in enumerate(self.keys):
            if i > 0:
                self.file.write(",")
            v = kvs.get(k)
            if v is not None:
                self.file.write(str(v))
        self.file.write("\n")
        self.file.flush()

    def close(self):
        self.file.close()


class TensorBoardOutputFormat(KVWriter):
    """
    Dumps key/value pairs into TensorBoard's numeric format.
    """

    def __init__(self, dir):
        os.makedirs(dir, exist_ok=True)
        self.step = 0
        self.writer = SummaryWriter(dir)

    def writekvs(self, kvs):
        for k, v in kvs.items():
            self.writer.add_scalar(k, v, self.step)

        self.writer.flush()

    def add_figure(self, tag, figure):
        self.writer.add_figure(tag, figure, self.step)

    def set_step(self, step):
        self.step = step

    def close(self):
        if self.writer:
            self.writer.Close()
            self.writer = None


def make_output_format(format, ev_dir, log_suffix=""):
    os.makedirs(ev_dir, exist_ok=True)
    if format == "stdout":
        return HumanOutputFormat(sys.stdout)
    elif format == "log":
        return HumanOutputFormat(osp.join(ev_dir, "experiment%s.log" % log_suffix))
    elif format == "json":
        return JSONOutputFormat(osp.join(ev_dir, "progress.json"))
    elif format == "csv":
        return CSVOutputFormat(osp.join(ev_dir, "progress.csv"))
    elif format == "tensorboard":
        return TensorBoardOutputFormat(ev_dir)
    else:
        raise ValueError("Unknown format specified: %s" % (format,))


# ================================================================
# API
# ================================================================


def logkv(key, val):
    """
    Log a value of some diagnostic
    Call this once for each diagnostic quantity, each iteration
    If called many times, last value will be used.
    """
    Logger.CURRENT.logkv(key, val)


def logkv_mean(key, val):
    """
    The same as logkv(), but if called many times, values averaged.
    """
    Logger.CURRENT.logkv_mean(key, val)


def logkvs(d):
    """
    Log a dictionary of key-value pairs
    """
    for (k, v) in d.items():
        logkv(k, v)


def set_tb_step(step):
    """
    record step for tensorboard
    """
    Logger.CURRENT.set_tb_step(step)


def add_figure(*args):
    """
    add_figure for tensorboard
    """
    Logger.CURRENT.add_figure(*args)


def dumpkvs():
    """
    Write all of the diagnostics from the current iteration

    level: int. (see logger.py docs) If the global logger level is higher than
                the level argument here, don't print to stdout.
    """
    Logger.CURRENT.dumpkvs()


def getkvs():
    return Logger.CURRENT.name2val


def log(*args, level=INFO):
    """
    Write the sequence of args, with no separators, to the console and output files (if you've configured an output file).
    """
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
    """
    Set logging threshold on current logger.
    """
    Logger.CURRENT.set_level(level)


def get_dir():
    """
    Get directory that log files are being written to.
    will be None if there is no output directory (i.e., if you didn't call start)
    """
    return Logger.CURRENT.get_dir()


record_tabular = logkv
record_step = set_tb_step
dump_tabular = dumpkvs


class ProfileKV:
    """
    Usage:
    with logger.ProfileKV("interesting_scope"):
        code
    """

    def __init__(self, n):
        self.n = "wait_" + n

    def __enter__(self):
        self.t1 = time.time()

    def __exit__(self, type, value, traceback):
        Logger.CURRENT.name2val[self.n] += time.time() - self.t1


def profile(n):
    """
    Usage:
    @profile("my_func")
    def my_func(): code
    """

    def decorator_with_name(func):
        def func_wrapper(*args, **kwargs):
            with ProfileKV(n):
                return func(*args, **kwargs)

        return func_wrapper

    return decorator_with_name


# ================================================================
# Backend
# ================================================================


class Logger(object):
    DEFAULT = None  # A logger with no output files. (See right below class definition)
    # So that you can still log to the terminal without setting up any output files
    CURRENT = None  # Current logger being used by the free functions above

    def __init__(self, dir, output_formats, precision=None):
        self.name2val = OrderedDict()
        self.level = INFO
        self.dir = dir
        self.output_formats = output_formats
        self.precision = precision  # float

    # Logging API, forwarded
    # ----------------------------------------
    def logkv(self, key, val):
        if self.precision is not None and isinstance(val, float):
            self.name2val[key] = round(val, self.precision)
        else:
            self.name2val[key] = val

    def add_figure(self, *args):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.add_figure(*args)

    def set_tb_step(self, step):
        for fmt in self.output_formats:
            if isinstance(fmt, TensorBoardOutputFormat):
                fmt.set_step(step)

    def dumpkvs(self):
        if self.level == DISABLED:
            return
        for fmt in self.output_formats:
            if isinstance(fmt, KVWriter):
                fmt.writekvs(self.name2val)
        self.name2val.clear()

    def log(self, *args, level=INFO):
        if self.level <= level:
            self._do_log(args)

    # Configuration
    # ----------------------------------------
    def set_level(self, level):
        self.level = level

    def get_dir(self):
        return self.dir

    def close(self):
        for fmt in self.output_formats:
            fmt.close()

    # Misc
    # ----------------------------------------
    def _do_log(self, args):
        for fmt in self.output_formats:
            if isinstance(fmt, SeqWriter):
                fmt.writeseq(map(str, args))


Logger.DEFAULT = Logger.CURRENT = Logger(
    dir=None, output_formats=[HumanOutputFormat(sys.stdout)]
)


def configure(dir=None, format_strs=None, log_suffix="", precision=None):
    if dir is None:
        dir = os.getenv("OPENAI_LOGDIR")
    if dir is None:
        dir = osp.join(
            tempfile.gettempdir(), datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        )

    assert isinstance(dir, str)
    os.makedirs(dir, exist_ok=True)

    if format_strs is None:
        strs = os.getenv("OPENAI_LOG_FORMAT")
        format_strs = strs.split(",") if strs else LOG_OUTPUT_FORMATS
    output_formats = [make_output_format(f, dir, log_suffix) for f in format_strs]

    Logger.CURRENT = Logger(dir=dir, output_formats=output_formats, precision=precision)
    log("Logging to %s" % dir)
