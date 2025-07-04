# XrRV code taken from xarray-einstats (Apache 2.0 license)
"""
The XrContinuousRV class is a wrapper for scipy.stats.rv_continuous
It calls the scipy.stats.rv_continuous.rvs method using xarray.apply_ufunc and returns xarray.DataArray

Here we're using it to create a class that automatically handles the weights of a mixture model.
"""


from collections.abc import Sequence

import numpy as np
import xarray as xr
import xclim as xc


class _XrRV:
    """Base random variable wrapper class.

    Most methods have a common signature between continuous and
    discrete variables in scipy. We define a base wrapper and
    then subclass it to add the specific methods like pdf or pmf.

    Notes
    -----
    One of the main goals of this library is ease of maintenance.
    We could wrap each distribution to preserve call signatures
    and avoid different behaviour between passing input arrays
    as args or kwargs, but so far we don't consider what we'd won
    doing this to be worth the extra maintenance burden.
    """

    def __init__(self, dist, *args, **kwargs):
        self.dist = dist
        self.args = args
        self.kwargs = kwargs

    def _broadcast_args(self, args, kwargs):
        """Broadcast and combine initialization and method provided args and kwargs."""
        len_args = len(args) + len(self.args)
        all_args = [*args, *self.args, *kwargs.values(), *self.kwargs.values()]
        broadcastable = []
        non_broadcastable = []
        b_idx = []
        n_idx = []
        for i, a in enumerate(all_args):
            if isinstance(a, xr.DataArray):
                broadcastable.append(a)
                b_idx.append(i)
            else:
                non_broadcastable.append(a)
                n_idx.append(i)
        broadcasted = list(xr.broadcast(*broadcastable))
        all_args = [
            x
            for x, _ in sorted(
                zip(broadcasted + non_broadcastable, b_idx + n_idx),
                key=lambda pair: pair[1],
            )
        ]
        all_keys = list(kwargs.keys()) + list(self.kwargs.keys())
        args = all_args[:len_args]
        kwargs = dict(zip(all_keys, all_args[len_args:]))
        return args, kwargs

    def rvs(
        self, *args, size=1, random_state=None, dims=None, apply_kwargs=None, **kwargs
    ):
        """Implement base rvs method.

        In scipy, rvs has a common signature that doesn't depend on continuous
        or discrete, so we can define it here.
        """
        args, kwargs = self._broadcast_args(args, kwargs)
        size_in = tuple()
        dims_in = tuple()
        for a in (*args, *kwargs.values()):
            if isinstance(a, xr.DataArray):
                size_in = a.shape
                dims_in = a.dims
                break

        if isinstance(dims, str):
            dims = [dims]

        if isinstance(size, (Sequence, np.ndarray)):
            if dims is None:
                dims = [f"rv_dim{i}" for i, _ in enumerate(size)]
            if len(dims) != len(size):
                raise ValueError("dims and size must have the same length")
            size = (*size, *size_in)
        elif size > 1:
            if dims is None:
                dims = ["rv_dim0"]
            if len(dims) != 1:
                raise ValueError("dims and size must have the same length")
            size = (size, *size_in)
        else:
            if size_in:
                size = size_in
            dims = None

        if dims is None:
            dims = tuple()

        if apply_kwargs is None:
            apply_kwargs = {}

        return xr.apply_ufunc(
            self.dist.rvs,
            *args,
            kwargs={**kwargs, "size": size, "random_state": random_state},
            input_core_dims=[dims_in for _ in args],
            output_core_dims=[[*dims, *dims_in]],
            **apply_kwargs,
        )


class XrContinuousRV(_XrRV):
    """Wrapper for subclasses of :py:class:`~scipy.stats.rv_continuous`.

    Usage examples available at :py:ref:`stats_tutorial`

    See Also
    --------
    xarray_einstats.stats.XrDiscreteRV

    Examples
    --------
    Evaluate the ppf of a Student-T distribution from DataArrays that need
    broadcasting:
    """


def _asdataarray(x_or_q, dim_name):
    """Ensure input is a DataArray.

    This is designed for the x or q arguments in univariate distributions.
    It is also used in multivariate normal distribution but only as a fallback.
    """
    if isinstance(x_or_q, xr.DataArray):
        return x_or_q
    x_or_q_ary = np.asarray(x_or_q)
    if x_or_q_ary.ndim == 0:
        return xr.DataArray(x_or_q_ary)
    if x_or_q_ary.ndim == 1:
        return xr.DataArray(
            x_or_q_ary, dims=[dim_name], coords={dim_name: np.asarray(x_or_q)}
        )
    raise ValueError(
        "To evaluate distribution methods on data with >=2 dims,"
        " the input needs to be a xarray.DataArray"
    )


def _wrap_method(method):
    def aux(self, *args, **kwargs):
        dim_name = "quantile" if method in {"ppf", "isf"} else "point"
        meth = getattr(self.dist, method)
        if args:
            args = (_asdataarray(args[0], dim_name), *args[1:])
        args, kwargs = self._broadcast_args(
            args, kwargs
        )  # pylint: disable=protected-access
        return xr.apply_ufunc(meth, *args, kwargs=kwargs, dask="parallelized")

    return aux


def _add_documented_method(cls, wrapped_cls, methods, extra_docs=None):
    """Register methods to XrRV classes and document them from a template."""
    if extra_docs is None:
        extra_docs = {}
    for method_name in methods:
        extra_doc = extra_docs.get(method_name, "")
        if method_name == "rvs":
            if wrapped_cls == "rv_generic":
                continue
            method = cls.rvs
        else:
            method = _wrap_method(method_name)
        setattr(
            method,
            "__doc__",
            f"Method wrapping :meth:`scipy.stats.{wrapped_cls}.{method_name}` "
            "with :func:`xarray.apply_ufunc`\n\nUsage examples available at "
            f":ref:`stats_tutorial/dists`.\n\n{extra_doc}",
        )
        setattr(cls, method_name, method)


doc_extras = {
    "rvs": """
Parameters
----------
args : scalar or array_like, optional
    Passed to the scipy distribution after broadcasting.
size : int of sequence of ints, optional
    The number of samples to draw *per array element*. If the distribution
    parameters broadcast to a ``(4, 10, 6)`` shape and ``size=(5, 3)`` then
    the output shape is ``(5, 3, 4, 10, 6)``. This differs from the scipy
    implementation. Here, all broadcasting and alignment is done for you,
    you give the dimensions the right names, and broadcasting just happens.
    If ``size`` followed scipy behaviour, you'd be forced to broadcast
    to provide a valid value which would defeat the ``xarray_einstats`` goal
    of handling all alignment and broadcasting for you.
random_state : optional
    Passed as is to the wrapped scipy distribution
dims : str or sequence of str, optional
    Dimension names for the dimensions created due to ``size``. If present
    it must have the same length as ``size``.
apply_kwargs : dict, optional
    Passed to :func:`xarray.apply_ufunc`
kwargs : dict, optional
    Passed to the scipy distribution after broadcasting using the same key.
"""
}
base_methods = ["cdf", "logcdf", "sf", "logsf", "ppf", "isf", "rvs"]
_add_documented_method(XrContinuousRV, "rv_generic", base_methods, doc_extras)
_add_documented_method(
    XrContinuousRV, "rv_continuous", base_methods + ["pdf", "logpdf"], doc_extras
)

# -------------- End of xarray-einstats code -------------- #


class XMixtureDistribution:
    """Statistical distribution based on a mixture of distributions."""

    def __init__(self, params, weights):
        """Create MixtureModel instance.

        Parameters
        ----------
        params: DataArray
            Distribution parameters returned by the `xclim.indices.stats.fit` function.
        weights: DataArray, or None
            Weights to apply to each distribution. Should share at least one dimension with `params`.
        """
        dist = xc.indices.stats.get_dist(params.attrs["scipy_dist"])

        # Create the XrContinuousRV instance and store it in self.xr
        self.xr = XrContinuousRV(dist, **xp2dict(params))

        # Store the weights
        self.weights = weights

        # Dimension for the application of weights
        self.dims = weights.dims

    @classmethod
    def from_data(cls, data, weights, distribution, method="ML"):
        """Return a MixtureModel instance from sample data.

        Parameters
        ----------
        data: xr.DataArray
            Sample with a `time` dimension along which the data will be fitted.
        weights: list of float
            List of weights to apply to each submodel.
        distribution: str
            Name of the distribution to fit, from scipy.stats continuous distributions.
        """
        # Fit data for each realization - dask-aware
        params = xc.indices.stats.fit(
            data, dist=distribution, dim="time", method=method
        )
        return cls(params=params, weights=weights)


def _wrap_mixture_method(method):
    """Convert XrRV method to apply to mixture models."""

    def aux(self, *args, **kwargs):
        meth = getattr(self.xr, method)
        out = meth(*args, **kwargs)
        out.name = method

        return out.weighted(self.weights).mean(dim=self.dims)

    return aux


for method in ["cdf", "logcdf", "sf", "logsf", "ppf", "isf", "pdf", "logpdf"]:
    setattr(XMixtureDistribution, method, _wrap_mixture_method(method))


def xp2dict(da):
    names = list(da["dparams"].data)
    return {name: da.sel(dparams=name) for name in names}
