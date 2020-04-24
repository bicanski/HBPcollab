import numpy as np

from scipy import stats


def tsplot(ax, data, **kw):
    """Time series plot to replace the deprecated seaborn function.

    :param ax:
    :param data:
    :param kw:
    :return:
    """
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    sd = np.std(data, axis=0)
    cis = (est - sd, est + sd)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def tsplot_boot(ax, data, **kw):
    """Plot time series with bootstrapped confidence intervals.

    :param ax:
    :param data:
    :param kw:
    :return:
    """
    x = np.arange(data.shape[1])
    est = np.mean(data, axis=0)
    cis = _bootstrap(data)
    ax.fill_between(x, cis[0], cis[1], alpha=0.2, **kw)
    ax.plot(x, est, **kw)
    ax.margins(x=0)


def _bootstrap(data, n_boot=10000, ci=68):
    """Helper function for tsplot_boot. Bootstraps confidence intervals for plotting time series.

    :param data:
    :param n_boot:
    :param ci:
    :return:
    """
    boot_dist = []
    for i in range(int(n_boot)):
        resampler = np.random.randint(0, data.shape[0], data.shape[0])
        sample = data.take(resampler, axis=0)
        boot_dist.append(np.mean(sample, axis=0))
    b = np.array(boot_dist)
    s1 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. - ci / 2.)
    s2 = np.apply_along_axis(stats.scoreatpercentile, 0, b, 50. + ci / 2.)
    return s1, s2
