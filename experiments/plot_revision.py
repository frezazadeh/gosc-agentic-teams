"""Figures, tables and statistics for the revision experiments.

    python -m experiments.plot_revision
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.plot import FIG, RES, STYLE, fmt  # noqa: E402  (shared style)

W1 = 3.45  # IEEE single column (in)
W2 = 7.0  # IEEE double column (in)
RNG = np.random.default_rng(0)
NBOOT = 4000

RULES = {
    "gosc": dict(color="#2a78d6", marker="o", label="Theory-of-mind VoI (GOSC)"),
    "gosc_tv": dict(color="#4a3aa7", marker="X", label="Belief-divergence value"),
    "gosc_bits": dict(color="#008300", marker="P", label="Throughput value (bits)"),
    "gosc_aoi": dict(color="#e34948", marker="v", label="Age-of-information value"),
    "sem_agg": dict(color="#eb6834", marker="s", label="Periodic aggregated, interval $k$",
                    linestyle="--"),
}
RULE_NAMES = {"gosc": "Theory-of-mind VoI (GOSC)", "gosc_tv": "Belief divergence",
              "gosc_bits": "Throughput (bits)", "gosc_aoi": "Age of information",
              "sem_agg": "Periodic aggregated"}


def load(name):
    with open(os.path.join(RES, f"rev_{name}.json")) as f:
        return json.load(f)["rows"]


def table(rows, key="completion_time"):
    """{(scheme, param): array over seeds (sorted by seed)}"""
    out = {}
    for r in rows:
        out.setdefault((r["scheme"], r["param"]), {})[r["seed"]] = r
    return {k: [v[s] for s in sorted(v)] for k, v in out.items()}


def arr(rs, key):
    return np.array([r[key] for r in rs], dtype=float)


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def paired(a, b):
    """mean(a-b) with 95% bootstrap CI over seeds, and #seeds where a>b."""
    d = np.asarray(a, float) - np.asarray(b, float)
    idx = np.random.default_rng(0).integers(0, len(d), size=(NBOOT, len(d)))
    boots = d[idx].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def fmt_ci(m, lo, hi, d=1):
    return f"{m:+.{d}f} [{lo:+.{d}f}, {hi:+.{d}f}]"


# ------------------------------------------------------------------------------
def frontier():
    tab = table(load("frontier"))
    curves = {}
    for (s, p), rs in tab.items():
        W, knob = p.split("/")
        curves.setdefault((W, s), []).append((knob, rs))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=True)
    stats = {}
    for ax, W in zip(axes, ("W100", "W50")):
        for s, st in RULES.items():
            pts = curves.get((W, s), [])
            if not pts:
                continue
            xs = [arr(rs, "symbols").mean() / 1e3 for _, rs in pts]
            ys = [arr(rs, "completion_time").mean() for _, rs in pts]
            order = np.argsort(xs)
            ax.plot(np.array(xs)[order], np.array(ys)[order], **st)
        sem = curves.get((W, "sem"))
        if sem:
            rs = sem[0][1]
            ax.plot([arr(rs, "symbols").mean() / 1e3], [arr(rs, "completion_time").mean()],
                    linestyle="none", marker="D", color="#eda100", label="Periodic semantic (default setting)")
        ax.set_xscale("log")
        ax.set_title(f"$W={W[1:]}$ channel uses per slot")
        ax.set_xlabel("Channel uses per mission ($\\times 10^3$, log scale)")
        stats[W] = matched(curves, W)
        stats[W]["periodic"] = best_periodic_vs_gosc(tab, W)
    axes[0].set_ylabel("Completion time (slots)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, -0.28))
    fig.savefig(os.path.join(FIG, "fig_frontier.pdf"))
    plt.close(fig)
    return stats


def _interp_T(pts, budget, idx):
    """Completion time at a channel-use budget, by log-linear interpolation along a
    frontier, using the seed resample idx.  None if the budget is outside the range."""
    xs, ys = [], []
    for _, rs in pts:
        sym = arr(rs, "symbols")[idx].mean()
        T = arr(rs, "completion_time")[idx].mean()
        xs.append(sym)
        ys.append(T)
    xs, ys = np.array(xs), np.array(ys)
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    if budget < xs[0] or budget > xs[-1]:
        return None
    # lower envelope: at each budget, best time achievable with at most that budget
    env = np.minimum.accumulate(ys)
    return float(np.interp(np.log(budget), np.log(xs), env))


def matched(curves, W):
    """Completion time of each rule at common channel-use budgets, with paired
    (same seed resample) bootstrap CIs of the difference to GOSC."""
    res = {}
    n = len(curves[(W, "gosc")][0][1])
    budgets = (5e3, 1e4, 2e4)
    for B in budgets:
        row = {}
        base_pt = _interp_T(curves[(W, "gosc")], B, np.arange(n))
        for s in RULES:
            if s == "gosc" or (W, s) not in curves:
                continue
            pt = _interp_T(curves[(W, s)], B, np.arange(n))
            if pt is None or base_pt is None:
                row[s] = None
                continue
            diffs = []
            rng = np.random.default_rng(0)
            for _ in range(10000):
                idx = rng.integers(0, n, n)
                a, b = _interp_T(curves[(W, s)], B, idx), _interp_T(curves[(W, "gosc")], B, idx)
                if a is not None and b is not None:
                    diffs.append(a - b)
            lo, hi = np.percentile(diffs, [2.5, 97.5])
            # Bonferroni over the 15 matched-budget comparisons with throughput/AoI values
            blo, bhi = np.percentile(diffs, [100 * 0.025 / 15, 100 - 100 * 0.025 / 15])
            row[s] = {"T": pt, "gosc_T": base_pt, "diff": pt - base_pt, "lo": float(lo), "hi": float(hi),
                      "bonf_lo": float(blo), "bonf_hi": float(bhi), "n_boot": len(diffs)}
        res[int(B)] = row
    return res


def best_periodic_vs_gosc(tab, prefix):
    """Best reporting interval of the periodic aggregated baseline vs GOSC at eta=0:
    paired difference in completion time and ratio of channel uses."""
    ks = [k for k in (1, 2, 4, 8, 16) if ("sem_agg", f"{prefix}/k{k}") in tab]
    best = min(ks, key=lambda k: arr(tab[("sem_agg", f"{prefix}/k{k}")], "completion_time").mean())
    a, g = tab[("sem_agg", f"{prefix}/k{best}")], tab[("gosc", f"{prefix}/eta0.0")]
    out = {"k": best, "diff": paired(arr(a, "completion_time"), arr(g, "completion_time")),
           "sym_ratio": float(arr(a, "symbols").mean() / arr(g, "symbols").mean()),
           "periodic_T": float(arr(a, "completion_time").mean()),
           "gosc_T": float(arr(g, "completion_time").mean())}
    if ("gosc_bits", f"{prefix}/eta0.0") in tab:
        bb = tab[("gosc_bits", f"{prefix}/eta0.0")]
        out["diff_vs_bits"] = paired(arr(a, "completion_time"), arr(bb, "completion_time"))
    return out


def frontier_table(stats):
    lines = []
    for W in ("W100", "W50"):
        for B, row in stats[W].items():
            if B == "periodic":
                continue
            g = next((v["gosc_T"] for v in row.values() if v), None)
            if g is None:
                continue
            cells = []
            for s in ("gosc_tv", "gosc_bits", "gosc_aoi"):
                v = row.get(s)
                cells.append("--" if not v else f"{v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]")
            lines.append(f"{W[1:]} & {B // 1000}k & {fmt(g) if g else '--'} & " + " & ".join(cells) + r" \\")
        if W == "W100":
            lines.append(r"\midrule")
    with open(os.path.join(FIG, "table_frontier.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")


def where_tom():
    """Frontiers where a decision-aware value could plausibly matter: a larger team
    (K=10) and clustered survivors."""
    tab = table(load("where_tom"))
    curves = {}
    for (s, p), rs in tab.items():
        W, knob = p.split("/")
        curves.setdefault((W, s), []).append((knob, rs))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.9), sharey=True)
    stats = {}
    for ax, W, title in zip(axes, ("K10", "clustered"), ("$K=10$ agents", "Clustered survivors")):
        for s, st in RULES.items():
            if s == "gosc_tv" or (W, s) not in curves:
                continue
            pts = curves[(W, s)]
            xs = [arr(rs, "symbols").mean() / 1e3 for _, rs in pts]
            ys = [arr(rs, "completion_time").mean() for _, rs in pts]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], **st)
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Channel uses per mission ($\\times 10^3$, log scale)")
        stats[W] = matched(curves, W)
        stats[W]["periodic"] = best_periodic_vs_gosc(tab, W)
    axes[0].set_ylabel("Completion time (slots)")
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.28))
    fig.savefig(os.path.join(FIG, "fig_where_tom.pdf"))
    plt.close(fig)
    lines = []
    for W, name in (("K10", "$K=10$"), ("clustered", "Clustered")):
        for B, row in stats[W].items():
            if B == "periodic":
                continue
            g = next((v["gosc_T"] for v in row.values() if v), None)
            if g is None:
                continue
            cells = []
            for sch in ("gosc_bits", "gosc_aoi"):
                v = row.get(sch)
                cells.append("--" if not v else f"{v['diff']:+.1f} [{v['lo']:+.1f}, {v['hi']:+.1f}]")
            lines.append(f"{name} & {B // 1000}k & {fmt(g) if g else '--'} & " + " & ".join(cells) + r" \\")
        if W == "K10":
            lines.append(r"\midrule")
    with open(os.path.join(FIG, "table_where_tom.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return stats


# ------------------------------------------------------------------------------
def ablation():
    tab = table(load("ablation"))
    names = {"gosc": "GOSC (ToM VoI + filter)", "gosc_nofilter": "ToM VoI, no filter",
             "gosc_bits": "Throughput value + filter", "gosc_novoi": "Throughput value, no filter",
             "gosc_nouep": "ToM VoI + filter, conventional link adaptation"}
    lines, out = [], {}
    for W in ("W100", "W50"):
        base = tab[("gosc", W)]
        for s in names:
            rs = tab[(s, W)]
            m, h = ci(arr(rs, "completion_time"))
            sym = arr(rs, "symbols").mean() / 1e3
            if s == "gosc":
                dcell = "--"
            else:
                d = paired(arr(rs, "completion_time"), arr(base, "completion_time"))
                dcell = fmt_ci(*d)
            out[f"{W}/{s}"] = {"T": m, "T_ci": h, "ksym": sym}
            lab = W[1:] if s == "gosc" else ""
            nm = r"\textbf{" + names[s] + "}" if s == "gosc" else names[s]
            lines.append(f"{lab} & {nm} & {fmt(m)} $\\pm$ {fmt(h)} & {fmt(sym, 1)} & {dcell} \\\\")
        if W == "W100":
            lines.append(r"\midrule")
    with open(os.path.join(FIG, "table_ablation.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def private():
    tab = table(load("private"))
    lines, out = [], {}
    names = {"default": "Default", "W50": "$W=50$", "snr-5": r"$\bar\gamma_{\mathrm{ref}}=-5$\,dB"}
    for sc in ("default", "W50", "snr-5"):
        sh, orc = tab[("gosc", f"{sc}/shared")], tab[("gosc", f"{sc}/oracle")]
        ptv = arr(sh, "private_tv").mean()
        ms, hs = ci(arr(sh, "completion_time"))
        mo, ho = ci(arr(orc, "completion_time"))
        d = paired(arr(sh, "completion_time"), arr(orc, "completion_time"))
        out[sc] = {"private_tv": ptv, "shared_T": ms, "oracle_T": mo, "diff": d,
                   "shared_ksym": arr(sh, "symbols").mean() / 1e3,
                   "oracle_ksym": arr(orc, "symbols").mean() / 1e3}
        lines.append(f"{names[sc]} & {ptv:.2f} & {fmt(ms)} $\\pm$ {fmt(hs)} & {fmt(mo)} $\\pm$ {fmt(ho)} & "
                     f"{fmt_ci(*d)} \\\\")
    with open(os.path.join(FIG, "table_private.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def downlink():
    tab = table(load("downlink"))
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.85), sharey=True)
    out = {}
    for ax, kind, vals, xl in ((axes[0], "loss", (0.0, 0.1, 0.3, 0.5), "Broadcast loss probability $p_{\\mathrm{DL}}$"),
                               (axes[1], "delay", (0, 1, 3, 5, 10), "Broadcast delay $d_{\\mathrm{DL}}$ (slots)")):
        for s in ("sem", "gosc", "genie"):
            st = dict(STYLE[s])
            key = lambda v: f"loss{v}" if kind == "loss" else ("loss0.0" if v == 0 else f"delay{v}")
            mu, hw = zip(*(ci(arr(tab[(s, key(v))], "completion_time")) for v in vals))
            mu, hw = np.array(mu), np.array(hw)
            ax.fill_between(vals, mu - hw, mu + hw, color=st["color"], alpha=0.12, linewidth=0)
            ax.plot(vals, mu, **st)
            for v in vals:
                rs = tab[(s, key(v))]
                out[f"{s}/{key(v)}"] = {"T": float(arr(rs, "completion_time").mean()),
                                         "ul_ksym": float(arr(rs, "symbols").mean() / 1e3),
                                         "dl_ksym": float(arr(rs, "dl_symbols").mean() / 1e3),
                                         "completed": float(arr(rs, "completed").mean())}
        ax.set_xlabel(xl)
        ax.set_ylim(0, 120)
    axes[0].set_ylabel("Completion time (slots)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.05, -0.28))
    fig.savefig(os.path.join(FIG, "fig_downlink.pdf"))
    plt.close(fig)
    # paired differences sem - gosc at each setting
    for v in ("loss0.0", "loss0.1", "loss0.3", "loss0.5", "delay1", "delay3", "delay5", "delay10"):
        out[f"diff/{v}"] = paired(arr(tab[("sem", v)], "completion_time"), arr(tab[("gosc", v)], "completion_time"))
    return out


def fbl():
    tab = table(load("fbl"))
    snrs = (-10, -5, 0, 5, 10, 15, 20)
    fig, axes = plt.subplots(1, 2, figsize=(W2, 1.85), sharey=True)
    out = {}
    for ax, g in zip(axes, (0.0, 2.0)):
        for s in ("raw", "nl", "sem", "gosc"):
            st = dict(STYLE[s])
            mu, hw = zip(*(ci(arr(tab[(s, f"gap{g}/snr{v}")], "completion_time")) for v in snrs))
            mu, hw = np.array(mu), np.array(hw)
            ax.fill_between(snrs, mu - hw, mu + hw, color=st["color"], alpha=0.12, linewidth=0)
            ax.plot(snrs, mu, **st)
            for v in snrs:
                rs = tab[(s, f"gap{g}/snr{v}")]
                out[f"gap{g}/{s}/snr{v}"] = {"T": float(arr(rs, "completion_time").mean()),
                                             "ksym": float(arr(rs, "symbols").mean() / 1e3),
                                             "completed": float(arr(rs, "completed").mean())}
        ax.set_title("FBL, no coding gap" if g == 0 else f"FBL, coding gap $\\Delta={g:.0f}$ dB")
        ax.set_xlabel("Reference SNR at 100 m (dB)")
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("Completion time (slots)")
    axes[0].legend(frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(1.05, -0.28))
    fig.savefig(os.path.join(FIG, "fig_fbl.pdf"))
    plt.close(fig)
    for g in (0.0, 2.0):
        for v in snrs:
            out[f"diff/gap{g}/snr{v}"] = paired(arr(tab[("sem", f"gap{g}/snr{v}")], "completion_time"),
                                                arr(tab[("gosc", f"gap{g}/snr{v}")], "completion_time"))
    return out


# ------------------------------------------------------------------------------
def main_paired():
    """Paired differences for the main comparisons (default operating point)."""
    with open(os.path.join(RES, "default.json")) as f:
        rows = json.load(f)["rows"]
    tab = {}
    for r in rows:
        tab.setdefault(r["scheme"], {})[r["seed"]] = r
    tab = {s: [v[k] for k in sorted(v)] for s, v in tab.items()}
    g = arr(tab["gosc"], "completion_time")
    return {s: paired(arr(rs, "completion_time"), g) for s, rs in tab.items() if s != "gosc"}


def rate_aware():
    """Strong external baseline: rate-aware periodic aggregated scheduler (best
    interval k per setting) vs GOSC at eta=0 and at the default price."""
    ra = table(load("rate_aware"))
    fr = table(load("frontier"))
    wt = table(load("where_tom"))
    src = {"W100": fr, "W50": fr, "K10": wt, "clustered": wt}
    out, lines = {}, []
    names = {"W100": "Default ($W=100$)", "W50": "$W=50$", "K10": "$K=10$", "clustered": "Clustered"}
    for st in ("W100", "W50", "K10", "clustered"):
        ks = [k for k in (1, 2, 4, 8) if ("sem_ra", f"{st}/k{k}") in ra]
        pts = {k: ra[("sem_ra", f"{st}/k{k}")] for k in ks}
        best = min(ks, key=lambda k: arr(pts[k], "completion_time").mean())
        b = pts[best]
        row = {"k": best, "T": float(arr(b, "completion_time").mean()),
               "ksym": float(arr(b, "symbols").mean() / 1e3),
               "all": {k: (float(arr(v, "completion_time").mean()), float(arr(v, "symbols").mean() / 1e3))
                       for k, v in pts.items()}}
        for eta in ("0.0", "1e-05"):
            g = src[st][("gosc", f"{st}/eta{eta}")]
            row[f"gosc_eta{eta}"] = {"T": float(arr(g, "completion_time").mean()),
                                    "ksym": float(arr(g, "symbols").mean() / 1e3),
                                    "diff": paired(arr(b, "completion_time"), arr(g, "completion_time")),
                                    "sym_ratio": float(arr(b, "symbols").mean() / arr(g, "symbols").mean())}
        out[st] = row
        g0, g1 = row["gosc_eta0.0"], row["gosc_eta1e-05"]
        lines.append(f"{names[st]} & {best} & {fmt(row['T'])} & {fmt(row['ksym'], 1)} & "
                     f"{fmt_ci(*g0['diff'])} & {g0['sym_ratio']:.2f} & {fmt_ci(*g1['diff'])} & {g1['sym_ratio']:.2f} \\\\")
    with open(os.path.join(FIG, "table_rate_aware.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def _curve(tab, scheme, prefix):
    return [(p, rs) for (s, p), rs in tab.items() if s == scheme and p.startswith(prefix + "/")]


def _resource_for(pts, metric, target, higher, idx):
    """Smallest channel use (on the best-achievable envelope, log-linear interpolation)
    at which the mean `metric` reaches `target`; None if never reached."""
    xs = np.array([arr(rs, "symbols")[idx].mean() for _, rs in pts])
    ys = np.array([arr(rs, metric)[idx].mean() for _, rs in pts])
    o = np.argsort(xs)
    xs, ys = xs[o], ys[o]
    env = np.maximum.accumulate(ys) if higher else np.minimum.accumulate(ys)
    ok = env >= target if higher else env <= target
    if not ok.any():
        return None
    i = int(np.argmax(ok))
    if i == 0:
        return float(xs[0])
    x0, x1, y0, y1 = np.log(xs[i - 1]), np.log(xs[i]), env[i - 1], env[i]
    f = 0.0 if y1 == y0 else (target - y0) / (y1 - y0)
    return float(np.exp(x0 + f * (x1 - x0)))


def add_periodic_best(tab, prefix, filt_prefix):
    """Merge the rate-aware periodic scheduler with and without the semantic filter
    into one 'best periodic' frontier (all intervals of both variants)."""
    ft = table(load("filtered_periodic"))
    for (s, p), rs in list(tab.items()):
        if s == "sem_ra" and p.startswith(prefix + "/"):
            tab[("periodic", p + "/nf")] = rs
    for (s, p), rs in ft.items():
        if p.startswith(filt_prefix + "/"):
            tab[("periodic", prefix + "/" + p.split("/", 1)[1] + "/f")] = rs
    try:
        sp = table(load("suppress_periodic"))
    except FileNotFoundError:
        sp = {}
    for (s, p), rs in sp.items():
        if p.startswith(filt_prefix + "/"):
            tab[("periodic", prefix + "/" + p.split("/", 1)[1] + "/s")] = rs
    return tab


def efficiency(tab, prefix, metric, target, higher, rules=("gosc", "gosc_bits", "gosc_aoi", "periodic"),
               nboot=2000):
    """Channel uses needed by each rule to reach `target`, and the ratio of the best
    periodic configuration to each value-priced rule (paired seed bootstrap).

    Failures are treated symmetrically: every resample is classified as both methods
    meeting the target, only the value-priced rule, only the periodic family, or neither.
    The ratio interval is conditional on both meeting the target, and the four
    frequencies are reported with it."""
    curves = {r: _curve(tab, r, prefix) for r in rules}
    n = len(next(iter(curves.values()))[0][1])
    full = np.arange(n)
    need = {r: _resource_for(c, metric, target, higher, full) for r, c in curves.items()}
    out = {"target": target, "need": need}
    rng = np.random.default_rng(0)
    for r in rules[:-1]:
        ratios, cats = [], {"both": 0, "only_rule": 0, "only_periodic": 0, "neither": 0}
        for _ in range(nboot):
            idx = rng.integers(0, n, n)
            a = _resource_for(curves[rules[-1]], metric, target, higher, idx)
            g = _resource_for(curves[r], metric, target, higher, idx)
            if a is not None and g is not None:
                cats["both"] += 1
                ratios.append(a / g)
            elif g is not None:
                cats["only_rule"] += 1
            elif a is not None:
                cats["only_periodic"] += 1
            else:
                cats["neither"] += 1
        ratios = np.array(ratios, float)
        point = None if need[r] is None or need[rules[-1]] is None else need[rules[-1]] / need[r]
        out[f"ratio_periodic_over_{r}"] = [point,
                                           float(np.percentile(ratios, 2.5)) if ratios.size else None,
                                           float(np.percentile(ratios, 97.5)) if ratios.size else None,
                                           cats["both"] / nboot]
        out[f"feasibility_{r}"] = {k: v / nboot for k, v in cats.items()}
    return out


def rescue():
    tab = table(load("rescue"))
    fig, axes = plt.subplots(1, 3, figsize=(W2, 1.85), sharey=True)
    out = {}
    styles = dict(RULES)
    styles["gosc"] = dict(RULES["gosc"], label="GOSC, theory-of-mind value")
    styles["gosc_bits"] = dict(RULES["gosc_bits"], label="GOSC transport, throughput value")
    styles["gosc_aoi"] = dict(RULES["gosc_aoi"], label="GOSC transport, AoI value")
    styles["sem_ra"] = dict(color="#eb6834", marker="s", label="Periodic, rate-aware ($k$ swept)",
                            linestyle="none")
    ftab = table(load("filtered_periodic"))
    for ax, (st, title) in zip(axes, (("W100", "$W=100$"), ("W50", "$W=50$"),
                                      ("snr-5", "$\\bar\\gamma_{\\mathrm{ref}}=-5$ dB"))):
        for r in ("gosc", "gosc_bits", "gosc_aoi", "sem_ra"):
            pts = _curve(tab, r, st)
            xs = [arr(rs, "symbols").mean() / 1e3 for _, rs in pts]
            ys = [100 * arr(rs, "saved_fraction").mean() for _, rs in pts]
            o = np.argsort(xs)
            ax.plot(np.array(xs)[o], np.array(ys)[o], **styles[r])
        sp = table(load("suppress_periodic"))
        env = _envelope([v for (sch, p), v in sp.items() if p.startswith("rescue-" + st + "/")],
                        "saved_fraction", True, 100.0)
        ax.plot([x for x, _ in env], [y for _, y in env], color="#4a3aa7", marker="X", linestyle="-",
                label="Periodic, suppressing (best envelope)")
        fp = [(p, v) for (sch, p), v in ftab.items() if p.startswith("rescue-" + st + "/")]
        ax.plot([arr(v, "symbols").mean() / 1e3 for _, v in fp],
                [100 * arr(v, "saved_fraction").mean() for _, v in fp], linestyle="none", marker="D",
                color="#eda100", label="Periodic, rate-aware + filter ($k$ swept)")
        for r, mk, col in (("sem", "P", "#4a3aa7"), ("raw", "^", "#1baf7a"), ("report", "v", "#e87ba4")):
            rs = tab[(r, f"{st}/ref")]
            ax.plot([max(arr(rs, "symbols").mean() / 1e3, 0.3)], [100 * arr(rs, "saved_fraction").mean()],
                    linestyle="none", marker=mk, color=col, label=STYLE[r]["label"])
        g = tab[("genie", f"{st}/ref")]
        ax.axhline(100 * arr(g, "saved_fraction").mean(), color="#7a7974", linestyle="--", linewidth=1.0,
                   label="Ideal-communication reference (genie)")
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel("Channel uses ($\\times 10^3$, log)")
        genie_saved = arr(g, "saved_fraction").mean()
        out[st] = {"genie_saved": float(genie_saved),
                   "refs": {r: {"saved": float(arr(tab[(r, f"{st}/ref")], "saved_fraction").mean()),
                                "T": float(arr(tab[(r, f"{st}/ref")], "completion_time").mean()),
                                "ksym": float(arr(tab[(r, f"{st}/ref")], "symbols").mean() / 1e3)}
                            for r in ("sem", "raw", "nl", "report", "genie")},
                   "points": {f"{r}/{p}": {"saved": float(arr(rs, "saved_fraction").mean()),
                                            "T": float(arr(rs, "completion_time").mean()),
                                            "ksym": float(arr(rs, "symbols").mean() / 1e3)}
                              for r in ("gosc", "gosc_bits", "gosc_aoi", "sem_ra")
                              for p, rs in _curve(tab, r, st)}}
        add_periodic_best(tab, st, "rescue-" + st)
        for frac in (0.90, 0.95, 1.0):
            out[st][f"eff_{int(frac * 100)}"] = efficiency(tab, st, "saved_fraction", frac * genie_saved, True)
    axes[0].set_ylabel("Survivors saved (%)")
    axes[0].legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(1.65, -0.28), fontsize=6.5)
    fig.savefig(os.path.join(FIG, "fig_rescue.pdf"))
    plt.close(fig)
    return out


def search_efficiency():
    """Same efficiency metric on the search task: channel uses to finish within a
    margin of the genie's completion time."""
    out = {}
    for name, W in (("frontier", "W100"), ("frontier", "W50"), ("where_tom", "K10"), ("where_tom", "clustered")):
        tab = table(load(name))
        ra = table(load("rate_aware"))
        tab.update({k: v for k, v in ra.items() if k[1].startswith(W + "/")})
        add_periodic_best(tab, W, W)
        # genie reference for the setting (the genie does not use the uplink, so its
        # time does not depend on W)
        g = table(load("genie_ref"))
        genie_T = float(arr(g[("genie", "W100" if W in ("W100", "W50") else W)], "completion_time").mean())
        out[W] = {f"within_{m}": efficiency(tab, W, "completion_time", genie_T * (1 + m / 100), False)
                  for m in (10, 15, 25)}
        out[W]["reference_T"] = genie_T
    return out


def _envelope(pts, metric, higher, scale=1.0):
    """Best-achievable envelope (x = mean channel uses in k, y = mean outcome) over a
    family of operating points."""
    xy = sorted((arr(v, "symbols").mean() / 1e3, scale * arr(v, metric).mean()) for v in pts)
    env, best = [], None
    for x, y in xy:
        if best is None or (y > best if higher else y < best):
            best = y
            env.append((x, y))
    return env


def efficiency_outputs(summary):
    """Main-result figure and tables: resource efficiency at matched outcomes."""
    # ---- figure: search (default) and rescue (default) frontiers
    fr = table(load("frontier"))
    ra = table(load("rate_aware"))
    rs = table(load("rescue"))
    ft = table(load("filtered_periodic"))
    fig, ax0 = plt.subplots(figsize=(W1, 2.1))
    panels = ((ax0, fr, ra, "W100", "W100", "completion_time", "Completion time (slots)",
               "Search task ($W=100$)", 1.0),)
    lab = {"gosc": "GOSC, theory-of-mind value", "gosc_bits": "GOSC transport, throughput value",
           "gosc_aoi": "GOSC transport, AoI value"}
    for ax, tab, ratab, pre, fpre, metric, ylab, title, scale in panels:
        for r in ("gosc", "gosc_bits", "gosc_aoi"):
            pts = _curve(tab, r, pre)
            xs = np.array([arr(v, "symbols").mean() / 1e3 for _, v in pts])
            ys = np.array([scale * arr(v, metric).mean() for _, v in pts])
            o = np.argsort(xs)
            st = dict(RULES[r]); st["label"] = lab[r]
            ax.plot(xs[o], ys[o], **st)
        for name, t2, pp, mk, col, lb in (("sem_ra", ratab, pre, "s", "#eb6834", "Periodic, rate-aware ($k$ swept)"),
                                           ("sem_raf", ft, fpre, "D", "#eda100",
                                            "Periodic, rate-aware + filter ($k$ swept)")):
            pts = _curve(t2, name, pp)
            xs = np.array([arr(v, "symbols").mean() / 1e3 for _, v in pts])
            ys = np.array([scale * arr(v, metric).mean() for _, v in pts])
            o = np.argsort(xs)
            ax.plot(xs[o], ys[o], linestyle="none", marker=mk, color=col, label=lb)
        sp = table(load("suppress_periodic"))
        pts = [v for (sch, p), v in sp.items() if p.startswith(fpre + "/")]
        env = _envelope(pts, metric, metric != "completion_time", scale)
        ax.plot([x for x, _ in env], [y for _, y in env], color="#4a3aa7", marker="X", linestyle="-",
                label="Periodic, suppressing (best envelope over $k$, $\\theta$)")
        ax.set_xscale("log")
        ax.set_xlabel("Uplink channel uses per mission ($\\times 10^3$, log scale)")
        ax.set_ylabel(ylab)
        ax.set_title(title)
    g = table(load("genie_ref"))[("genie", "W100")]
    ax0.axhline(arr(g, "completion_time").mean(), color="#7a7974", linestyle="--", linewidth=1.0,
                label="Ideal-communication reference (genie)")
    ax0.legend(frameon=False, ncol=1, loc="upper center", bbox_to_anchor=(0.5, -0.30), fontsize=6.0)
    fig.savefig(os.path.join(FIG, "fig_efficiency.pdf"))
    plt.close(fig)

    # ---- efficiency table
    def cell(e, r):
        v = e.get(f"ratio_periodic_over_{r}")
        if not v or v[0] is None:
            return "--"
        lo = "--" if v[1] is None else f"{v[1]:.1f}"
        hi = "--" if v[2] is None else f"{v[2]:.1f}"
        return f"{v[0]:.1f} [{lo}, {hi}] ({100 * v[3]:.0f}\\%)"

    def need(e, r):
        n = e["need"].get(r)
        return "--" if n is None else f"{n / 1e3:.1f}"

    def censored(tab, r, pre, e):
        """True if the target is already met at the rule's lowest tested channel use,
        so its true requirement may be lower (the ratio is then a lower bound)."""
        n = e["need"].get(r)
        pts = _curve(tab, r, pre)
        return n is not None and pts and abs(n - min(arr(v, "symbols").mean() for _, v in pts)) < 1e-6

    def cellc(e, r, tab, pre):
        c = cell(e, r)
        return ("$\\ge$" + c.split(" [")[0] + " [" + c.split(" [")[1]) if c != "--" and censored(tab, r, pre, e) else c

    lines = []
    se = summary["search_efficiency"]
    for W, name in (("W100", "Search"), ("W50", "Search, $W{=}50$"), ("K10", "Search, $K{=}10$"),
                    ("clustered", "Search, clustered")):
        for m in ("within_15", "within_25"):
            e = se[W][m]
            tgt = f"$T\\le{1 + int(m.split('_')[1]) / 100:.2f}\\,T_{{\\mathrm{{g}}}}$"
            stab = table(load("frontier")) if W in ("W100", "W50") else table(load("where_tom"))
            lines.append(f"{name} & {tgt} & {need(e, 'periodic')} & {need(e, 'gosc')} & {cellc(e, 'gosc', stab, W)} & "
                         f"{cellc(e, 'gosc_bits', stab, W)} \\\\")
    lines.append("\\midrule")
    for st, name in (("W100", "Rescue"), ("W50", "Rescue, $W{=}50$"), ("snr-5", "Rescue, $-5$\\,dB")):
        for tg in ("eff_90", "eff_100"):
            e = summary["rescue"][st][tg]
            frac = tg.split("_")[1]
            tgt = f"$S\\ge{int(frac) / 100:.2f}\\,S_{{\\mathrm{{g}}}}$"
            lines.append(f"{name} & {tgt} & {need(e, 'periodic')} & {need(e, 'gosc')} & {cellc(e, 'gosc', rs, st)} & "
                         f"{cellc(e, 'gosc_bits', rs, st)} \\\\")
    with open(os.path.join(FIG, "table_efficiency.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")

    # ---- value-fidelity table
    rows = json.load(open(os.path.join(RES, "rev_value_fidelity.json")))
    rng = np.random.default_rng(0)

    def rk(x):
        return np.argsort(np.argsort(x))

    def spear(x, y):
        x, y = np.asarray(x, float), np.asarray(y, float)
        r = np.corrcoef(rk(x), rk(y))[0, 1]
        bs = []
        for _ in range(2000):
            i = rng.integers(0, len(x), len(x))
            if np.std(x[i]) > 0 and np.std(y[i]) > 0:
                bs.append(np.corrcoef(rk(x[i]), rk(y[i]))[0, 1])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return f"{r:+.2f} [{lo:+.2f}, {hi:+.2f}]"

    fid, lines = {}, []
    names = {"tom": "Theory-of-mind VoI", "tv": "Belief divergence", "bits": "Throughput (bits)",
             "aoi": "Age of information"}
    E = [r for r in rows if r["kind"] == "E"]
    EI = [r for r in rows if r["kind"] == "EI"]
    cl = json.load(open(os.path.join(RES, "rev_fidelity_analysis.json")))["corr_cluster"]
    f3 = lambda v: f"{v[0]:+.2f} [{v[1]:+.2f}, {v[2]:+.2f}]"
    for e, nm in names.items():
        a, b, c = f3(cl[f"E/{e}/gt_gain"]), f3(cl[f"E/{e}/dT"]), f3(cl[f"EI/{e}/dT"])
        lines.append(f"{nm} & {a} & {b} & {c} \\\\")
        fid[e] = (a, b, c)
    with open(os.path.join(FIG, "table_fidelity.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    fid["n"] = {"E": len(E), "EI": len(EI), "I": sum(1 for r in rows if r["kind"] == "I")}
    fid["changed"] = {k: float(np.mean([r["gt_gain"] != 0 for r in rows if r["kind"] == k])) for k in ("E", "I", "EI")}
    fid["mean_dT"] = {k: float(np.mean([r["dT"] for r in rows if r["kind"] == k])) for k in ("E", "I", "EI")}
    return fid


def joint_summary():
    """Exact set-valued (joint) scheduling vs independent valuation (paired)."""
    tab = table(load("joint"))
    out = {}
    for p in sorted({k[1] for k in tab}):
        g = tab[("gosc", p)]
        for sch in ("gosc_joint", "gosc_joint_nb"):
            j = tab[(sch, p)]
            out[f"{sch}/{p}"] = {"dT": paired(arr(j, "completion_time"), arr(g, "completion_time")),
                                 "sym_ratio": float(arr(j, "symbols").mean() / arr(g, "symbols").mean())}
    return out


def validation_table():
    """Fresh-seed validation of frozen operating points, with paired outcome differences
    and each method's margin to the common target (bootstrap over the fresh seeds)."""
    v = json.load(open(os.path.join(RES, "rev_validation.json")))
    names = {"search-W100": "Search", "search-K10": "Search, $K{=}10$", "search-clustered": "Search, clustered",
             "rescue-W100": "Rescue", "rescue-W50": "Rescue, $W{=}50$", "rescue-snr-5": "Rescue, $-5$\\,dB"}
    rng = np.random.default_rng(0)
    lines, out = [], {}
    for k, nm in names.items():
        r = v[k]
        time = r["metric"] == "completion_time"
        sc = 1.0 if time else 100.0
        g = np.array(r["gosc"]["metric_per_seed"]) * sc
        a = np.array(r["periodic"]["metric_per_seed"]) * sc
        tgt = r["target"] * sc
        idx = rng.integers(0, len(g), (4000, len(g)))

        def ci(x):
            b = x[idx].mean(1)
            return float(x.mean()), *[float(q) for q in np.percentile(b, [2.5, 97.5])]
        # margin: positive = target met with room (time: target - T; saved: S - target)
        mg = ci((tgt - g) if time else (g - tgt))
        ma = ci((tgt - a) if time else (a - tgt))
        d = ci(g - a)                      # GOSC minus periodic outcome, paired
        out[k] = {"margin_gosc": mg, "margin_periodic": ma, "diff": d, "ratio": r["ratio"],
                  "gosc_point": r["gosc"]["point"], "periodic_point": r["periodic"]["point"],
                  "periodic_scheme": r["periodic"]["scheme"], "target": tgt,
                  "gosc_outcome": float(g.mean()), "periodic_outcome": float(a.mean()),
                  "gosc_ksym": r["gosc"]["fresh_ksym"], "periodic_ksym": r["periodic"]["fresh_ksym"]}
        f = lambda x: f"{x[0]:+.1f} [{x[1]:+.1f}, {x[2]:+.1f}]"
        lines.append(f"{nm} & {tgt:.1f} & {g.mean():.1f} & {f(mg)} & {r['gosc']['fresh_ksym']:.1f} & "
                     f"{a.mean():.1f} & {f(ma)} & {r['periodic']['fresh_ksym']:.1f} & {f(d)} & "
                     f"{r['ratio'][0]:.2f} [{r['ratio'][1]:.2f}, {r['ratio'][2]:.2f}] \\\\")
        if k == "search-clustered":
            lines.append("\\midrule")
    with open(os.path.join(FIG, "table_validation.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    # frozen configurations (appendix)
    conf = []
    for k, nm in names.items():
        r = v[k]
        gp = {"1e-05": "10^{-5}", "3e-05": "3\\cdot10^{-5}", "0.0001": "10^{-4}", "0.0003": "3\\cdot10^{-4}",
              "3e-06": "3\\cdot10^{-6}", "0.001": "10^{-3}", "0.0": "0"}[r["gosc"]["point"].split("eta")[1]]
        pp = r["periodic"]["point"].split("/", 1)[1]
        sch = {"sem_ra": "rate-aware", "sem_raf": "rate-aware + filter", "sem_ras": "suppressing"}[r["periodic"]["scheme"]]
        pp = pp.replace("k", "$k{=}").replace("/th", "$, $\\theta{=}") + "$"
        conf.append(f"{nm} & $\\eta={gp}$ & {sch}, {pp} \\\\")
    with open(os.path.join(FIG, "table_frozen.tex"), "w") as fh:
        fh.write("\n".join(conf) + "\n")
    return out


def sweep_paired():
    """Paired sem - GOSC differences and channel-use ratios along the main sweeps."""
    out = {}
    for name in ("snr", "budget", "agents"):
        with open(os.path.join(RES, f"{name}.json")) as f:
            tab = table(json.load(f)["rows"])
        for p in sorted({k[1] for k in tab}):
            a, g = tab[("sem", p)], tab[("gosc", p)]
            out[f"{name}/{p}"] = list(paired(arr(a, "completion_time"), arr(g, "completion_time"))) + [
                float(arr(a, "symbols").mean() / arr(g, "symbols").mean())]
    return out


def completion_rates():
    """Settings where some missions hit the horizon (capped completion time)."""
    out = {}
    for name in ("snr", "budget", "agents"):
        with open(os.path.join(RES, f"{name}.json")) as f:
            rows = json.load(f)["rows"]
        agg = {}
        for r in rows:
            agg.setdefault((r["scheme"], r["param"]), []).append(r["completed"])
        for (s, p), v in agg.items():
            if np.mean(v) < 1:
                out[f"{name}/{s}/{p}"] = float(np.mean(v))
    return out


def merge_value_tables():
    """One matched-budget table over all settings (was two tables).
    Columns: setting, budget, GOSC, belief divergence, throughput, AoI."""
    def rows(fname):
        with open(os.path.join(FIG, fname)) as f:
            return [ln.rstrip() for ln in f if ln.strip() and not ln.startswith(r"\midrule")]
    out = []
    for ln in rows("table_frontier.tex"):          # W & budget & GOSC & TV & bits & AoI
        c = ln.split("&")
        out.append(f"$W={c[0].strip()}$ & " + "&".join(c[1:]))
    out.append(r"\midrule")
    for ln in rows("table_where_tom.tex"):         # setting & budget & GOSC & bits & AoI
        c = [x.strip() for x in ln.split("&")]
        c[-1] = c[-1].replace(r"\\", "").strip()
        out.append(f"{c[0]} & {c[1]} & {c[2]} & n/e & {c[3]} & {c[4]} " + r"\\")
    with open(os.path.join(FIG, "table_values.tex"), "w") as f:
        f.write("\n".join(out) + "\n")


def main():
    summary = {}
    for name, fn in (("frontier", frontier), ("where_tom", where_tom), ("ablation", ablation), ("private", private),
                     ("downlink", downlink), ("fbl", fbl)):
        try:
            summary[name] = fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"skipping {name}: {e!r}")
    if "frontier" in summary:
        frontier_table(summary["frontier"])
    summary["rate_aware"] = rate_aware()
    for name, fn in (("rescue", rescue), ("search_efficiency", search_efficiency)):
        try:
            summary[name] = fn()
        except (FileNotFoundError, KeyError) as e:
            print(f"skipping {name}: {e!r}")
    summary["fidelity"] = efficiency_outputs(summary)
    summary["main_paired"] = main_paired()
    summary["sweep_paired"] = sweep_paired()
    summary["joint"] = joint_summary()
    summary["validation"] = validation_table()
    summary["fidelity_cluster"] = json.load(open(os.path.join(RES, "rev_fidelity_analysis.json")))
    summary["completion_rates"] = completion_rates()
    merge_value_tables()
    sched = os.path.join(RES, "rev_scheduler.json")
    if os.path.exists(sched):
        with open(sched) as f:
            summary["scheduler"] = json.load(f)
    with open(os.path.join(RES, "summary_revision.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("revision figures and tables written")


if __name__ == "__main__":
    main()
