import numpy as np
from scipy import stats


def genomic_lambda(pvals):
    """基因组膨胀因子 λ_GC = median(观测 χ²) / median(χ²_1df 理论)。"""
    p = np.asarray(pvals, dtype=float)
    chi2_obs = stats.chi2.isf(p, df=1)
    chi2_med_expected = stats.chi2.ppf(0.5, df=1)
    return float(np.median(chi2_obs) / chi2_med_expected)


def qq_points(pvals):
    """QQ 图数据：返回 (期望 -log10p, 观测 -log10p)，均升序。"""
    p = np.sort(np.array(pvals, dtype=float))
    n = len(p)
    expected = -np.log10(np.arange(1, n + 1) / (n + 1))
    observed = -np.log10(p)
    return expected[::-1], observed[::-1]
