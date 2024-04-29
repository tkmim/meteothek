# main program of xrboots
# contatins xrboots class and functions to do actual calculation

import logging
import numpy as np
import xarray as xr
from typing import List, Union
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import meteothek


# set up the logger and set the logging level
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class xrboots:
    """
    A class for bootstrapping for specified dimensions on an xr.Dataset object.

    Parameters:
        dataset (xr.Dataset): xarray.Dataset which will be bootstrapped

    """

    def __init__(self, dataset: xr.Dataset, dims: list, iterations: int, seed: int):
        """
        Initializes the xrboots library.

        Parameters:
            dataset (xr.Dataset): The dataset on which the calculations or operations will be performed.
        """
        self.dataset = dataset

    @staticmethod
    def boot1d(dataset: xr.Dataset, dim: str, iterations: int, n: int = 4, seed: int = 123, *, replace: bool = True):
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
        n : int (default=4)
            The number of samples to draw for each iteration.
        seed : int (default=123)
            The seed value for random number generation.

        Returns
        -------
        xr.Dataset
            The bootstrapped dataset.

        """
        np.random.seed(seed=seed)
        CONCAT_KWARGS = {"coords": "minimal", "compat": "override"}

        # generate a random list of indices
        ibclists = np.random.choice(dataset[dim].size, n * iterations, replace=replace).reshape((n, iterations))
        logger.debug(seed, ibclists[:, 66])

        forecast_smp_list: List[Union[xr.Dataset, xr.DataArray]] = []

        for ii in np.arange(iterations):
            ibcidx = ibclists[:, ii]
            forecast_smp_list.append(dataset.isel({dim: ibcidx}).assign_coords({dim: np.arange(n)}))
        forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
        forecast_smp["iteration"] = np.arange(iterations)

        forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
        return forecast_smp.transpose(..., "iteration")

    @staticmethod
    def bootnd(dataset: xr.Dataset, dims: list, iterations: int, n: int, seed: int, *, replace: bool = True):
        """
        Perform bootstrapping along n dimensions in a dataset.

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
                            {
                                key: values[i] if values.size > 1 else values
                                for (key, values) in random_choices.items()
                            },
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
                for ii, dim in enumerate(dims):
                    random_choices[dim] = np.random.choice(dataset[dim].size, n ** (ii + 1), replace=replace).reshape(
                        [n] * (ii + 1)
                    )

                for key, value in random_choices.items():
                    logger.debug(key, value)

                forecast_smp_list.append(recursive_isel(dataset, dims, random_choices, n))

        # concatinating the bootstrapped datasets
        with meteothek.Timer("bootnd: concating"):
            forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
        forecast_smp["iteration"] = np.arange(iterations)

        forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
        return forecast_smp.transpose(..., "iteration")

    @staticmethod
    def bootnd_fixed(dataset: xr.Dataset, dims: list, iterations: int, n: int, seed: int, *, replace: bool = True):
        """
        Perform bootstrapping along n dimensions in a dataset, with fixed samples for each dimension within a single iteration.

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

        # list to store the bootstrapped datasets
        forecast_smp_list: List[Union[xr.Dataset, xr.DataArray]] = []

        # tqdm is used to show the progress bar
        with logging_redirect_tqdm(loggers=[logger]):
            for ii in tqdm(np.arange(iterations)):
                # generate a list of n random indices for each dimension
                random_choices = {}
                for dim in dims:
                    random_choices[dim] = np.random.choice(dataset[dim].size, n, replace=replace).reshape(n)

                for key, value in random_choices.items():
                    logger.debug(key, value)

                forecast_smp_list.append(
                    dataset.isel(**random_choices).assign_coords({key: np.arange(n) for key in dims})
                )

        # concatinating the bootstrapped datasets
        with meteothek.Timer("bootnd: concating"):
            forecast_smp = xr.concat(forecast_smp_list, dim="iteration", **CONCAT_KWARGS)
        forecast_smp["iteration"] = np.arange(iterations)

        # reordering the dimensions for better readability
        forecast_smp = forecast_smp[[*dataset.coords, "iteration", *dataset.variables]]
        return forecast_smp.transpose(..., "iteration")
