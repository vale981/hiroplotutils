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
from joblib import Parallel, delayed
import logging
import itertools

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


def make_figure(fig_name: str | None = None, *args, **kwargs):
    meta = get_function_meta()

    fig_name = fig_name or (meta[1] if isinstance(meta[1], str) else meta[1].name)
    fig_name += "".join(f"_{k}={str(v)}" for k, v in meta[2].items()) if meta[2] else ""

    fig = plt.figure(fig_name, *args, **kwargs)

    fig.__dict__["__hiro_filename_function"] = get_function_meta()
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


def get_function_meta():
    i = 1
    stack = inspect.stack()
    frame = stack[i]
    while "site-packages" in frame.filename or "hiroplotutils" in frame.filename:
        i += 1
        if i >= len(stack):
            frame = None
            break

        frame = stack[i]

    filename = pathlib.Path(frame.filename) if frame else "<unknown>"
    function = frame.function if frame else "<unknown>"

    return filename, function, get_kwargs(frame)


def write_meta(path, include_kwags=True, filename_function_override=None, **kwargs):
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

    filename, function, fn_kwargs = filename_function_override or get_function_meta()

    if filename != "<unknown>":
        filename = str(filename.relative_to(project_dir))

    outpath = f"{path}.meta.yaml"
    with open(outpath, "w") as f:
        yaml.dump(
            dict(
                source=filename,
                function=function,
                function_args=fn_kwargs if include_kwags else {},
                change_id=change_id,
                commit_id=commit_id,
                description=description.strip(),
                refers_to=str(path.relative_to(project_dir)),
            )
            | kwargs,
            f,
            allow_unicode=True,
        )

    print(f"Metadata written to {outpath}")


def get_kwargs(frame=None):
    frame = frame.frame or inspect.currentframe().f_back.f_back.f_back.f_back
    keys, _, _, values = inspect.getargvalues(frame)
    kwargs = {}
    for key in keys:
        if key != "self":
            kwargs[key] = values[key]

    return kwargs


@noop_if_interactive
def save_figure(
    fig, name, extra_meta=None, include_kwags=True, directory="./figs", *args, **kwargs
):
    import pickle

    directory = (pathlib.Path.cwd()) / directory
    directory.mkdir(exist_ok=True)

    plt.savefig(directory / f"{name}.pdf", *args, **kwargs)
    plt.savefig(directory / f"{name}.png", *args, dpi=600, **kwargs)

    print(f"Figure saved as {directory}/{name}.pdf")
    pickle_path = directory / f"{name}.pkl"

    with open(pickle_path, "wb") as f:
        pickle.dump(fig, f)

    write_meta(
        directory / f"{name}.pdf",
        name=name,
        include_kwags=include_kwags,
        extra_meta=extra_meta,
        filename_function_override=fig.__dict__.get("__hiro_filename_function", None),
    )


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


class PlotContainer:
    """A container for plots that can be executed in parallel."""

    def __init__(
        self,
        figdir: str | pathlib.Path,
        sizes: dict[str, float],
        default_ratio: float = 0.61803398876,
        pyplot_config: Callable | None = None,
    ):
        self._plots = []
        self._figdir = figdir
        self._sizes = sizes
        self._default_ratio: float = default_ratio
        self._pyplot_config = pyplot_config or (lambda: None)

    def _save_fig(self, fig, size: str | tuple[str, float], num: tuple[int, int]):
        horizontal = self._sizes[size[0] if isinstance(size, tuple) else size]
        vertical: float = size[1] if isinstance(size, tuple) else self._default_ratio

        fig.set_size_inches(horizontal, vertical * horizontal)
        save_figure(
            fig,
            f"{num[0]:03}_{num[1]:03}_" + fig.get_label(),
            directory=str(self._figdir),
        )

    def register(self, size: str | tuple[str, float], *args: dict):
        """Registers a plot to be executed."""

        if hasattr(sys, "ps1"):
            self._pyplot_config()
            return lambda f: f

        if len(args) == 1 and isinstance(args[0], Callable):
            return self.register(next(iter(self._sizes.keys())))(args[0])

        if len(args) == 0:
            args = ({},)

        def decorator(f):
            plots = []
            plot_index = len(self._plots) + 1
            for sub_index, keywords in enumerate(args):
                logging.info("Registered plot", f, keywords)
                plots.append(
                    (
                        lambda: [
                            self._pyplot_config(),
                            self._save_fig(
                                f(**keywords),
                                size,
                                (plot_index, sub_index + 1),
                            ),
                        ]
                        and None,
                        f,
                        keywords,
                    )
                )

            self._plots.append(plots)
            return f

        return decorator

    def execute_plots(self, *args, **kwargs):
        """Executes the given list of plots in parallel."""

        if hasattr(sys, "ps1"):
            return

        flattened = list(itertools.chain.from_iterable(self._plots))
        import argparse

        parser = argparse.ArgumentParser()
        parser.add_argument("--n_jobs", type=int, default=-1)
        parser.add_argument(
            "--list", help="List all available plots", action="store_true"
        )
        parser.add_argument(
            "--only",
            type=int,
            help="Only execute the plot with the given index.",
            default=None,
            choices=[i + 1 for i in range(len(flattened))],
            nargs="+",
        )
        cmd_args = parser.parse_args()

        if cmd_args.list:
            print("hi")
            format = "{:2d} {:03}_{:03}_{:<30} "
            total = 1
            for i, plot_group in enumerate(self._plots):
                for sub_index, plot in enumerate(plot_group):
                    print(
                        format.format(total, i + 1, sub_index + 1, plot[1].__name__),
                        end="",
                    )
                    print(" ".join(f"{k}={v}" for k, v in plot[2].items()))
                    total += 1

                print("")
            return

        only = (
            range(len(flattened))
            if cmd_args.only is None
            else [i - 1 for i in cmd_args.only]
        )

        if "n_jobs" not in kwargs or cmd_args.n_jobs != -1:
            kwargs["n_jobs"] = cmd_args.n_jobs

        if "backend" not in kwargs:
            kwargs["backend"] = "loky"

        Parallel(*args, **kwargs)(delayed(flattened[i][0])() for i in only)
