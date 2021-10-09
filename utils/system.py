import os
import numpy as np
import random
import torch
import datetime
import dateutil.tz


def reproduce(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # below makes Conv operators much slower though deterministic
        # https://github.com/pytorch/pytorch/issues/40134
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]
