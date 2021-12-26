import numpy as np
import random
import torch
import datetime
import dateutil.tz


def reproduce(seed):
    """
    This can only fix the randomness of numpy and torch
    To fix the environment's, please use
        env.seed(seed)
        env.action_space.np_random.seed(seed)
    We have add these in our training script
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def now_str():
    now = datetime.datetime.now(dateutil.tz.tzlocal())
    return now.strftime("%m-%d:%H-%M:%S.%f")[:-4]
