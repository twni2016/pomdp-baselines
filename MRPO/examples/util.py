import json

import numpy as np


class NumpyEncoder(json.JSONEncoder):
    """Ensures json.dumps doesn't crash on numpy types
    See: https://stackoverflow.com/questions/27050108/convert-numpy-type-to-python/27050186#27050186
    """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)
