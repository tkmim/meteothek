import logging
import numpy as np
import xarray as xr
from typing import List, Union
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from meteothek import Timer

# set up the logger and set the logging level
logger = logging.getLogger(__name__)


def boot1d(
    dataset: xr.Dataset,
    dim: str,
    n: int,
    iterations: int,
    seed: int = 123,
    *,
    replace: bool = True,
):
    """
    Perform bootstrapping along one dimension in a dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        The input Xarray dataset.
    dim : str
        The dimension along which to perform bootstrapping.
    iterations : int
        The number of bootstrapping iterations.
    n : int
        The number of samples to draw for each iteration.
    seed : int (default=123)
        The seed value for random number generation.

    Returns
    -------
    xr.Dataset
        The bootstrapped dataset.

    """

    if not isinstance(n, int):
        raise TypeError("Sample size 'n' must be an integer.")
    if not isinstance(iterations, int):
        raise TypeError("Number of iterations must be an integer.")

    np.random.seed(seed=seed)
    CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}

    # generate a random list of indices
    ibclists = np.random.choice(dataset[dim].size, n * iterations, replace=replace).reshape((n, iterations))
    logger.debug(f"{seed}, {ibclists[:, 66]}")

    forecast_smp_list: List[Union[xr.Dataset, xr.DataArray]] = []

    for ii in np.arange(iterations):
        ibcidx = ibclists[:, ii]
        forecast_smp_list.append(dataset.isel({dim: ibcidx}).assign_coords({dim: np.arange(n)}))
    forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)

    forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
    return forecast_smp.transpose(..., "iteration")


def bootnd(
    dataset: xr.core.types.T_Xarray,
    dims: list,
    ns: Union[int, list],
    iterations: int,
    seed: int = 123,
    *,
    replace: bool = True,
):
    """
    Perform bootstrapping along n dimensions in a dataset.

    This function employs a simple sampling method, with which sample clusters are chosen for each dimension and sample overlapping parts of the clusters. For example, if bootstrap is applied along dimensions A and B, both having n=5, and subsample size is 3 and 2 for each. The resulted sample will be 6 samples, which are indicated as in the following table.
    ```
    A\B | 1 | 2 | 3 | 4 | 5
    1   |   | x |   | x |
    2   |   |   |   |   |
    3   |   | x |   | x |
    4   |   | x |   | x |
    5   |   |   |   |   |
    ```

    This sampleing method represents a realistic situation that we fully utilise limited resources (e.g. using all combinations of ICs and LBCs), but the given samples would likely be correlated.

    Parameters
    ----------
    dataset : xr.Dataset
        The input Xarray dataset.
    dims : list[str,]
        a list of dimensions along which to perform bootstrapping.
    ns : list[int, ]
        a list of number of samples to draw for each iteration.
    iterations : int
        a number of bootstrapping iterations.
    seed : int (default=123)
        The seed value for random number generation.
    replace : bool (default=True)
        Whether to sample with replacement.


    Returns
    -------
    xr.Dataset
        The bootstrapped dataset.

    """

    # Check types of arguments
    # Check if dims is a list containing only strings
    if not isinstance(dims, list) or not all(isinstance(dim, str) for dim in dims):
        raise TypeError("dims must be a list containing only string variables.")

    np.random.seed(seed=seed)
    CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}

    # list to store the bootstrapped datasets
    forecast_smp_list: List[Union[xr.Dataset, xr.DataArray]] = []

    # tqdm is used to show the progress bar
    with logging_redirect_tqdm(loggers=[logger]):
        for ii in tqdm(np.arange(iterations)):
            # generate a list of n random indices for each dimension
            random_choices = {}
            for n, dim in zip(ns, dims):
                random_choices[dim] = np.random.choice(dataset[dim].size, n, replace=replace).reshape(n)

            forecast_smp_list.append(
                dataset.isel(**random_choices).assign_coords({key: np.arange(n) for n, key in zip(ns, dims)})
            )

    # concatinating the bootstrapped datasets
    with Timer("bootnd: concating"):
        forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)

    # reordering the dimensions for better readability
    forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]

    return forecast_smp.transpose(..., "iteration")


def bootnd_multistage(dataset: xr.Dataset, dims: list, iterations: int, n: int, seed: int, *, replace: bool = True):
    """
    **This function still requires mathematical justification.**

    Perform multi-stage random bootstrapping along n dimensions in a dataset.

    ## multi-stage random sampling
    Generating a subsample cluster for each sample recursively. For the k-th dimenstion (k > 1), n_k samples are randomly drawn for each sample for the (k-1)-th dimension.


    Parameters
    ----------
    dataset : xr.Dataset
        The input Xarray dataset.
    dims : list
        The list of dimensions along which to perform bootstrapping.
    iterations : int
        The number of bootstrapping iterations.
    n : int (default=4)
        The number of samples to draw for each iteration.
        Currently the same n for all dimensions is supported.
    seed : int (default=123)
        The seed value for random number generation.
    replace : bool (default=True)
        Whether to sample with replacement.


    Returns
    -------
    xr.Dataset
        The bootstrapped dataset.

    """

    np.random.seed(seed=seed)
    CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}

    def recursive_isel(dataset, dims, random_choices, n):
        # recursive function to isel the dataset and combine the sampled datasets
        if len(dims) == 1:
            return dataset.isel({dims[0]: random_choices[dims[0]]}).assign_coords({dims[0]: np.arange(n)})
        else:
            recursive_datasets = []
            for i in range(n):
                logger.debug(
                    n, {key: values[i] if values.size > 1 else values for (key, values) in random_choices.items()}
                )
                recursive_datasets.append(
                    recursive_isel(
                        dataset.isel({dims[0]: [random_choices[dims[0]][i]]}).assign_coords({dims[0]: [i]}),
                        dims[1:],
                        {key: values[i] if values.size > 1 else values for (key, values) in random_choices.items()},
                        n,
                    )
                )

            return xr.concat(recursive_datasets, dim=dims[0])

    # list to store the bootstrapped datasets
    forecast_smp_list: List[Union[xr.Dataset, xr.DataArray]] = []

    # tqdm is used to show the progress bar
    with logging_redirect_tqdm(loggers=[logger]):
        for ii in tqdm(np.arange(iterations)):
            # generate a list of random indices for each dimension
            # the number of random indices is n^(ii+1) for the ii-th dimension
            random_choices = {}
            for jj, dim in enumerate(dims):
                random_choices[dim] = np.random.choice(dataset[dim].size, n ** (jj + 1), replace=replace).reshape(
                    [n] * (jj + 1)
                )
                print(random_choices[dim].shape)

            for key, value in random_choices.items():
                logger.debug(key, value)

            forecast_smp_list.append(recursive_isel(dataset, dims, random_choices, n))

    # concatinating the bootstrapped datasets
    with Timer("bootnd: concating"):
        forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)

    forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
    return forecast_smp.transpose(..., "iteration")
