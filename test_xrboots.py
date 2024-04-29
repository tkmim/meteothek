import numpy as np
import xarray as xr
import logging
from meteothek.xrboots import xrboots

# set up the logger and set the logging level
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# generate a logger handler
# handlerの生成
stream_handler = logging.StreamHandler()
# handlerのログレベル設定(ハンドラが出力するエラーメッセージのレベル)
# set the log level displayed in the handler
stream_handler.setLevel(logging.DEBUG)
logger.addHandler(stream_handler)

# Get a reference to the 'xrboots' logger
lo = logging.getLogger('meteothek.xrboots.xrboots')

# Change the log level to DEBUG
lo.setLevel(logging.INFO)
lo.addHandler(stream_handler)


# Create a sample dataset
data = np.random.rand(10, 5, 7)
coords = {"time": np.arange(10), "space": np.arange(5), "ens": np.arange(7)}
dataset = xr.Dataset({"data": (["time", "space", "ens"], data)}, coords=coords)

# Create an instance of xrboots
dims = [
    "time",
]
iterations = 1000
seed = 123
n = 4

logger.debug(dataset["time"].size)
# Test the boot1d method
boot1d_result = xrboots.boot1d(dataset, dims[0], iterations, n, seed)
logger.debug(boot1d_result)


# Test the bootnd method
dims = ["space", "ens"]
bootnd_result = xrboots.bootnd(dataset, dims, iterations, n, seed)
logger.debug(bootnd_result)

# Assert that the bootnd result has the correct number of iterations
assert bootnd_result.sizes["iteration"] == iterations

# Assert that the bootnd result is transposed correctly
# assert bootnd_result.transpose(..., *dims, 'iteration').shape == (..., n, n, iterations)

# Test the bootnd method
dims = ["space", "ens"]
bootnd_result = xrboots.bootnd_fixed(dataset, dims, iterations, n, seed)
logger.debug(bootnd_result)


print("All tests passed!")
