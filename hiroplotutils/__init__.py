import matplotlib.pyplot as plt
from typing import Callable, Any
from functools import wraps
from typing_extensions import ParamSpec, TypeVar, Concatenate
import yaml
import inspect
import subprocess
import pathlib
import sys
import numpy as np


P = ParamSpec("P")
R = TypeVar("R")


def noop_if_interactive(f):
    @wraps(f)
    def wrapped(*args, **kwargs):
        if hasattr(sys, "ps1"):
            print("NOOP as the session is interactive!")
            return

        return f(*args, **kwargs)

    return wrapped


def make_figure(fig_name: str = "interactive", *args, **kwargs):
    fig = plt.figure(fig_name, *args, **kwargs)
    fig.clf()
    return fig


def wrap_plot(
    plot_function: Callable[Concatenate[plt.Figure | None, P], R],  # pyright: ignore [reportPrivateImportUsage]
) -> Callable[Concatenate[plt.Figure | None, P], tuple[plt.Figure, R]]:  # pyright: ignore [reportPrivateImportUsage]
    """Decorator to wrap a plot function to inject the correct figure
    for interactive use.  The function that this decorator wraps
    should accept the figure as first argument.

    :param fig_name: Name of the figure to create.  By default it is
        "interactive", so that one plot window will be reused.
    :param setup_function: Function that returns a figure to use.  If
        it is provided, the ``fig_name`` will be ignored.
    """

    def wrapped(fig, *args: P.args, **kwargs: P.kwargs):
        if fig is None:
            fig = make_figure()

        ret_val = plot_function(fig, *args, **kwargs)
        return (fig, ret_val)

    return wrapped


def autoclose(f):
    def wrapped(*args, **kwargs):
        plt.close()
        return f(*args, **kwargs)

    return wrapped


def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import matplotlib.colors as mc
    import colorsys

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def get_jj_info(type):
    return subprocess.run(
        ["jj", "log", "-T", type, "-n", "1", "--no-graph"],
        stdout=subprocess.PIPE,
    ).stdout.decode("utf-8")


def write_meta(path, include_kwags=True, **kwargs):
    """Write metatdata for result that has been written to a file
    under ``path``.

    The metadata includes the change_id, commit_id and the description
    of the current ``jj`` state, and the source file that generated
    the result.  Additional metadata can be provided through the
    keyword arguments.  If ``include_kwargs`` is set to True, the
    keyword arguments will of the calling function be included in the
    metadata.
    """
    change_id = get_jj_info("change_id")
    commit_id = get_jj_info("commit_id")
    description = get_jj_info("description")
    project_dir = (
        subprocess.run("git rev-parse --show-toplevel", shell=True, capture_output=True)
        .stdout.decode("utf-8")
        .strip()
    )

    frame = inspect.stack()[3]
    module = inspect.getmodule(frame[0])
    filename = str(
        pathlib.Path(module.__file__).relative_to(project_dir)  # type: ignore
        if module
        else "<unknown>"
    )
    function = frame.function

    outpath = f"{path}.meta.yaml"
    with open(outpath, "w") as f:
        yaml.dump(
            dict(
                source=filename,
                function=function,
                function_args=get_kwargs() if include_kwags else {},
                change_id=change_id,
                commit_id=commit_id,
                description=description.strip(),
                refers_to=str(path),
            )
            | kwargs,
            f,
            allow_unicode=True,
        )

    print(f"Metadata written to {outpath}")


def get_kwargs():
    frame = inspect.currentframe().f_back.f_back.f_back.f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]
    return kwargs


@noop_if_interactive
def save_figure(fig, name, extra_meta=None, include_kwags=True, *args, **kwargs):
    import pickle

    dir = pathlib.Path(f"./figs/")
    dir.mkdir(exist_ok=True)
    fig.tight_layout()

    write_meta(
        f"./figs/{name}.pdf",
        name=name,
        include_kwags=include_kwags,
        extra_meta=extra_meta,
    )

    plt.savefig(f"./figs/{name}.pdf", *args, **kwargs)
    plt.savefig(f"./figs/{name}.png", *args, dpi=600, **kwargs)

    print(f"Figure saved as ./figs/{name}.pdf")
    pickle_path = dir / f"{name}.pkl"

    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)


@noop_if_interactive
def quick_save_pickle(obj, name, include_kwags=False, **kwargs):
    """Quickly save an object to a pickle file with metadata."""
    import pickle

    path = pathlib.Path(f"./outputs/{name}.pkl")
    path.parent.mkdir(exist_ok=True)

    with open(path, "wb") as f:
        pickle.dump(obj, f)

    write_meta(path, include_kwags=include_kwags, **kwargs)


def quick_load_pickle(name):
    """Quickly load an object from a pickle file."""

    import pickle

    path = pathlib.Path(f"./outputs/{name}.pkl")

    with open(path, "rb") as f:
        return pickle.load(f)


def scientific_round(val, *err, retprec=False):
    """Scientifically rounds the values to the given errors."""
    val, err = np.asarray(val), np.asarray(err)
    if len(err.shape) == 1:
        err = np.array([err])
        err = err.T
    err = err.T

    if err.size == 1 and val.size > 1:
        err = np.ones_like(val) * err

    if len(err.shape) == 0:
        err = np.array([err])

    if val.size == 1 and err.shape[0] > 1:
        val = np.ones_like(err) * val

    i = np.floor(np.log10(err))
    first_digit = (err // 10**i).astype(int)
    prec = (-i + np.ones_like(err) * (first_digit <= 3)).astype(int)
    prec = np.max(prec, axis=1)

    def smart_round(value, precision):
        value = np.round(value, precision)
        if precision <= 0:
            value = value.astype(int)
        return value

    if val.size > 1:
        rounded = np.empty_like(val)
        rounded_err = np.empty_like(err)
        for n, (value, error, precision) in enumerate(zip(val, err, prec)):
            rounded[n] = smart_round(value, precision)
            rounded_err[n] = smart_round(error, precision)

        if retprec:
            return rounded, rounded_err, prec
        else:
            return rounded, rounded_err

    else:
        prec = prec[0]
        if retprec:
            return (smart_round(val, prec), *smart_round(err, prec)[0], prec)
        else:
            return (smart_round(val, prec), *smart_round(err, prec)[0])
