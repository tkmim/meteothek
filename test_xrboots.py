import numpy as np
import xarray as xr
import logging
from meteothek.xrboots import *

# set up the logger and set the logging level
logger = logging.getLogger(__name__)

# showin DEBUG level messages
logging.basicConfig(level=logging.DEBUG)  # , format='%(levelname)s - %(name)s - %(asctime)s - %(message)s')

# showing messages in and above the INFO level in pynus.decode
logging.getLogger("meteothek.xrboots.xrboots").setLevel(logging.DEBUG)

# Create a sample dataset
data = np.random.rand(10, 5, 7)
coords = {"time": np.arange(10), "space": np.arange(5), "ens": np.arange(7)}
dataset = xr.Dataset({"data": (["time", "space", "ens"], data)}, coords=coords)

# Create an instance of xrboots
dims = ["time"]
iterations = 1000
seed = 123
n = 2

# Test the boot1d method
boot1d_result = xrboots.boot1d(dataset, dims[0], n, iterations, seed)
logger.debug(boot1d_result)


# Test the bootnd method
dims = ["space", "ens"]
bootnd_result = xrboots.bootnd(dataset, dims, (3, 4), iterations, seed)
logger.debug(bootnd_result)

# Assert that the bootnd result has the correct number of iterations
assert bootnd_result.sizes["iteration"] == iterations

# Assert that the bootnd result is transposed correctly
# assert bootnd_result.transpose(..., *dims, 'iteration').shape == (..., n, n, iterations)

# Test the bootnd method
dims = ["space", "ens"]
# bootnd_result = bootnd_fixed(dataset, dims, iterations, n, seed)
# logger.debug(bootnd_result)


print("All tests passed!")
