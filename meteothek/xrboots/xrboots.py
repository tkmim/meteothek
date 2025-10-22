import logging
import numpy as np
import xarray as xr
from typing import List, Union
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
from meteothek import Timer

# set up the logger and set the logging level
logger = logging.getLogger(__name__)
logger.propagate = False


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


def bootnd_subblock(
    dataset: xr.core.types.T_Xarray,
    dims: list,
    ns: Union[int, list],
    iterations: int,
    seed: int = 123,
    *,
    replace: bool = True,
):
    r"""
    Perform bootstrapping along n dimensions in a dataset by constructing a sub block in perturbation space.

    This function employs a simple sampling method, with which sample perturbations are chosen for each dimension and a sub block in perturbation space is constructed.
    For example, if bootstrap is applied along dimensions A and B, both having n=5, and subsample size is 3 and 2 for each.
    One sub-sampled member thus shares the same perturbation in A with two other members, and shares the same perturbation in B with another member, forming a sub-block as illustrated below:
    ```
    A\B | 1 | 2 | 3 | 4 | 5      A\B | 2 | 4
    1   |   | x |   | x |        1   | x | x
    2   |   |   |   |   |        3   | x | x
    3   |   | x |   | x |     -> 4   | x | x
    4   |   | x |   | x |
    5   |   |   |   |   |
    ```

    This method corresponds to choosing number of input parameters for each dimension and constructing an ensemble by using all combinations of these parameters.
    This represents a situation that we fully utilise limited perturbations to make as large an ensemble as possible (e.g. you have 3 ICs and 4 LBCs and produce 12 members by using all combinations).
    The sub-sampled members may be systematically correlated.

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

    # concatenating the bootstrapped datasets
    with Timer("bootnd: concateng"):
        forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)

    # reordering the dimensions for better readability
    forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]

    return forecast_smp.transpose(..., "iteration")


def bootnd(dataset: xr.Dataset, dims: dict, iterations: int, seed: int, *, replace: bool = True):
    r"""

    Perform bootstrapping along n dimensions in a dataset.

    ## multi-stage random sampling
    Generating a subsample cluster for each sample recursively. For the k-th dimenstion (k > 1), n_k samples are randomly drawn for each sample for the (k-1)-th dimension.


    Parameters
    ----------
    dataset : xr.Dataset
        The input Xarray dataset.
    dims : dictionary
        The dictionary containing a dimension name as key and number of samples to draw for each iteration as value.
    iterations : int
        The number of bootstrapping iterations.
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

    def recursive_isel(dataset, dims, random_choices):
        # recursive function to isel the dataset and combine the sampled datasets
        if len(dims) == 1:
            (dim, n), = dims.items()
            return dataset.isel({dim: random_choices[dim]}).assign_coords({dim: np.arange(n)})
        else:
            ns = list(dims.values())
            dims = list(dims.keys())
            
            recursive_datasets = []
            for i in range(ns[0]):
                logdict = {key: values[i] if values.size > 1 else values for (key, values) in random_choices.items()}
                logger.debug(f"{logdict}")

                recursive_datasets.append(
                    recursive_isel(
                        dataset.isel({dims[0]: [random_choices[dims[0]][i]]}).assign_coords({dims[0]: [i]}),
                        {dim:n for dim, n in zip(dims[1:], ns[1:])},
                        {key: values[i] if values.ndim > 1 else values for (key, values) in random_choices.items()},
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
            tn = []
            for jj, (dim, n) in enumerate(dims.items()):
                tn.append(n)
                random_choices[dim] = np.random.choice(dataset[dim].size, np.prod(tn), replace=replace).reshape(tn)

            for key, value in random_choices.items():
                logger.debug(f"{key}, {value}")

            forecast_smp_list.append(recursive_isel(dataset, dims, random_choices))

    # concatenating the bootstrapped datasets
    with Timer("bootnd: concateng"):
        forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
    forecast_smp["iteration"] = np.arange(iterations)

    forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
    return forecast_smp.transpose(..., "iteration")
