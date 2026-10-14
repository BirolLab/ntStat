import numpy as np
import scipy.ndimage
import scipy.optimize
import scipy.special
import scipy.stats

err_dists = [
    (lambda p: scipy.stats.gamma(p[1], loc=p[0], scale=p[2]), 3),
    (lambda p: scipy.stats.expon(loc=p[0], scale=p[2]), 3),
]

genomic_dists = [
    (lambda p: scipy.stats.nbinom(p[0] * p[1] / (1 - p[1] + 1e-9), p[1] + 1e-9), 2),
    (lambda p: scipy.stats.norm(loc=p[0], scale=p[0] * p[1] + 1e-9), 2),
    (lambda p: scipy.stats.skewnorm(5, loc=p[0], scale=p[0] * p[1] + 1e-9), 2),
]


def update_components(params):
    num_components = (len(params) - 6) // 3
    err, num_err_params = err_dists[int(params[0])]
    genomic, num_rv_params = genomic_dists[int(params[1])]
    weights = [params[2]]
    rvs = [err(params[3 : num_err_params + 3])]
    for i in range(num_components):
        i_w = 6 + i * 3
        weights.append(params[i_w])
        rv_params = params[i_w + 1 : i_w + num_rv_params + 1]
        rvs.append(genomic(rv_params))
    sum_w = sum(weights)
    components = [(w / sum_w, rv) for w, rv in zip(weights, rvs)]
    return components


def score(x, components):
    scores = np.empty(shape=(len(components), x.shape[0]))
    for i, (w, rv) in enumerate(components):
        if isinstance(rv.dist, scipy.stats.rv_continuous):
            y = rv.cdf(x + 0.5) - rv.cdf(x - 0.5)
        else:
            y = rv.pmf(x)
        scores[i, :] = w * y
    return scores


def loss(params, x, y):
    fx = score(x, update_components(params)).sum(axis=0)
    return np.abs(np.sqrt(y) - np.sqrt(fx)).sum()


def log_iteration(result: scipy.optimize.OptimizeResult, history: list):
    components = update_components(result.x)
    history.append((components, result.fun))


class Model:
    def __init__(self) -> None:
        self.__components = []

    @property
    def converged(self) -> bool:
        return len(self.__components) > 0

    @property
    def err_rv(self):
        return self.__components[0]

    @property
    def heterozygous_rv(self):
        return self.__components[1]

    @property
    def homozygous_rv(self):
        return self.__components[2]

    @property
    def copy_rvs(self):
        return self.__components[3:]

    @property
    def coverage(self):
        return self.peaks[1]

    @property
    def peaks(self):
        return np.array([rv.mean() for _, rv in self.__components[1:]])

    def pdf(self, x):
        return self.score_components(x).sum(axis=0)

    def score_components(self, x):
        return score(x, self.__components)

    def get_responsibilities(self, x):
        scores = self.score_components(x)
        return np.nan_to_num(scores / (scores.sum(axis=0) + 1e-9))

    def get_weak_robust_crossover(self, x):
        scores = self.score_components(x)
        y1 = scores[0, :].reshape(-1)
        y2 = scores[1:, :].sum(axis=0).reshape(-1)
        i = np.where(y2 >= y1)[0]
        return x[i[0]] if i.shape[0] > 0 else 0

    def fit(self, hist, num_components=2, config=None):
        config = {} if config is None else config
        x, y = np.arange(1, hist.max_count + 1), hist.as_distribution()
        bounds = [
            (0, len(err_dists) - 1),
            (0, len(genomic_dists) - 1),
            (0, 1),
            (0, 0.999),
            (0.01, 1.0),
            (0.01, hist.max_count),
        ]
        x0 = [0, 0, 1, 0.9, 0.5, 0.5]
        smoothed_y = scipy.ndimage.gaussian_filter1d(y, sigma=10)
        peaks, _ = scipy.signal.find_peaks(smoothed_y[hist.first_minima :])
        if len(peaks) > 0:
            valid_peaks = peaks + hist.first_minima
            d = valid_peaks[smoothed_y[valid_peaks].argmax()] + 1
        else:
            d = smoothed_y[hist.first_minima :].argmax() + hist.first_minima + 1
        for i in range(num_components):
            if i == 0:
                b_mean = (d * 0.25, d * 0.75)
                x_mean = d * 0.5
            else:
                b_mean = (d * 0.8, d * 3)
                x_mean = d * (i + 1) / 2
            bounds.extend([(0, 1), b_mean, (0.1, 0.999)])
            x_mean_safe = max(b_mean[0] + 1e-5, min(x_mean, b_mean[1] - 1e-5))
            x0.extend([0.5, x_mean_safe, 0.5])
        history = []
        callback = lambda intermediate_result: log_iteration(
            intermediate_result,
            history,
        )
        opt = scipy.optimize.differential_evolution(
            func=loss,
            popsize=config.get("popsize", 3),
            init=config.get("init", "sobol"),
            x0=x0,
            bounds=bounds,
            integrality=[True, True] + [False] * (len(bounds) - 2),
            args=(x, y),
            seed=config.get("seed", 42),
            updating="deferred",
            workers=config.get("workers", -1),
            maxiter=config.get("maxiter", 5000),
            callback=callback,
            mutation=config.get("mutation", (0.5, 1.0)),
            recombination=config.get("recombination", 0.8),
            strategy=config.get("strategy", "best1bin"),
            polish=True,
        )
        components = update_components(opt.x)
        sorted_components = sorted(components[1:], key=lambda c: c[1].mean())
        self.__components = [components[0]] + list(sorted_components)
        return opt.nit, loss(opt.x, x, y), history
