"""Turn results/*.json into the paper's figures and tables.

    python -m experiments.plot
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

# Categorical slots in fixed order (validated palette); genie is a neutral reference.
STYLE = {
    "gosc":       dict(color="#2a78d6", marker="o", label="GOSC (proposed)"),
    "sem":        dict(color="#eb6834", marker="s", label="Semantic, periodic"),
    "raw":        dict(color="#1baf7a", marker="^", label="Raw sensor data"),
    "nl":         dict(color="#eda100", marker="D", label="Natural language"),
    "report":     dict(color="#e87ba4", marker="v", label="Report only"),
    "gosc_novoi": dict(color="#008300", marker="P", label="GOSC w/o VoI"),
    "gosc_tv":    dict(color="#4a3aa7", marker="X", label="GOSC, belief-div. VoI"),
    "genie":      dict(color="#7a7974", marker="", label="Ideal-communication reference (genie)", linestyle="--"),
}
NAMES = {
    "report": "Report only", "raw": "Raw sensor data", "nl": "Natural language (LLM-style)",
    "sem": "Semantic, periodic", "gosc_nouep": "GOSC w/o UEP", "gosc_novoi": "GOSC w/o VoI",
    "gosc_tv": "GOSC w/ belief-divergence VoI", "gosc": "GOSC (proposed)",
    "genie": "Ideal-comm. reference (genie)",
}
MAIN = ["report", "raw", "nl", "sem", "gosc", "genie"]

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 6.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#52514e", "axes.linewidth": 0.6,
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e4e3de", "grid.linewidth": 0.5,
    "lines.linewidth": 1.4, "lines.markersize": 4,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05, "pdf.fonttype": 42,
})
W1 = 3.45  # IEEE single column (in)
W2 = 7.0   # IEEE double column (in)


def load(name):
    with open(os.path.join(RES, f"{name}.json")) as f:
        return json.load(f)["rows"]


def ci(x):
    x = np.asarray(x, float)
    return x.mean(), 1.96 * x.std(ddof=1) / np.sqrt(len(x))


def by(rows, scheme, param=None):
    return [r for r in rows if r["scheme"] == scheme and (param is None or r["param"] == param)]


def summarize(rows):
    ct = [r["completion_time"] for r in rows]
    lat = [r["confirm_latency"] for r in rows if r["confirm_latency"] is not None]
    m, h = ci(ct)
    return {
        "T": m, "T_ci": h,
        "T_median": float(np.median(ct)),
        "completed": float(np.mean([r["completed"] for r in rows])),
        "kbits": float(np.mean([r["bits"] for r in rows])) / 1e3,
        "ksym": float(np.mean([r["symbols"] for r in rows])) / 1e3,
        "bits_per_found": float(np.mean([r["bits"] / max(r["found"], 1) for r in rows])),
        "false": float(np.mean([r["false_declarations"] for r in rows])),
        "latency": float(np.mean(lat)) if lat else float("nan"),
        "n": len(rows),
    }


def sweep_plot(name, xlabel, fname, xlog=False, xticks=None):
    rows = load(name)
    params = sorted({r["param"] for r in rows})
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for s in MAIN:
        st = dict(STYLE[s])
        mu, hw = zip(*(ci([r["completion_time"] for r in by(rows, s, p)]) for p in params))
        mu, hw = np.array(mu), np.array(hw)
        ax.fill_between(params, mu - hw, mu + hw, color=st["color"], alpha=0.12, linewidth=0)
        ax.plot(params, mu, **st)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Completion time (slots)")
    if xlog:
        ax.set_xscale("log", base=2)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
    ax.set_ylim(bottom=0)
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)
    return {s: {p: summarize(by(rows, s, p)) for p in params} for s in MAIN}


def sweep_bits_plot(name, xlabel, fname, xlog=False, xticks=None):
    rows = load(name)
    params = sorted({r["param"] for r in rows})
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    for s in MAIN:
        if s == "genie":
            continue  # ideal channel: no radio resources modelled
        st = dict(STYLE[s])
        ax.plot(params, [np.mean([r["symbols"] for r in by(rows, s, p)]) / 1e3 for p in params], **st)
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Channel uses ($\\times 10^3$)")
    if xlog:
        ax.set_xscale("log", base=2)
    if xticks is not None:
        ax.set_xticks(xticks)
        ax.set_xticklabels([str(x) for x in xticks])
    ax.legend(ncol=2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def sweep_pair_plot(name, xlabel, fname):
    """Two panels read together: completion time (left) and channel uses (right)."""
    rows = load(name)
    params = sorted({r["param"] for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(W2, 2.0))
    for s in MAIN:
        st = dict(STYLE[s])
        mu, hw = zip(*(ci([r["completion_time"] for r in by(rows, s, p)]) for p in params))
        mu, hw = np.array(mu), np.array(hw)
        axes[0].fill_between(params, mu - hw, mu + hw, color=st["color"], alpha=0.12, linewidth=0)
        axes[0].plot(params, mu, **st)
        if s == "genie":
            continue  # ideal channel: no radio resources modelled
        st2 = dict(st); st2.pop("label", None)
        axes[1].plot(params, [np.mean([r["symbols"] for r in by(rows, s, p)]) / 1e3 for p in params], **st2)
    axes[0].set_ylabel("Completion time (slots)")
    axes[0].set_ylim(bottom=0)
    axes[1].set_ylabel("Channel uses ($\\times 10^3$)")
    axes[1].set_yscale("log")
    for ax in axes:
        ax.set_xlabel(xlabel)
    axes[0].legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(1.1, -0.24))
    fig.savefig(os.path.join(FIG, fname))
    plt.close(fig)


def curve_plot():
    rows = load("default")
    fig, ax = plt.subplots(figsize=(W1, 2.0))
    T = 250
    for s in MAIN:
        st = dict(STYLE[s])
        c = np.mean([r["curve"][:T] for r in by(rows, s)], axis=0)
        st["markevery"] = 25
        ax.plot(np.arange(1, T + 1), c, **st)
    ax.set_xlabel("Time slot")
    ax.set_ylabel("Survivors located (of 8)")
    ax.set_xlim(0, T)
    ax.set_ylim(0, 8.2)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(os.path.join(FIG, "fig_curve.pdf"))
    plt.close(fig)


def pareto_plot():
    rows = load("pareto")
    tv = load("pareto_tv")
    dflt = load("default")
    fig, ax = plt.subplots(figsize=(W1, 2.5))

    def front(data, scheme):
        etas = sorted({r["param"] for r in data})
        return etas, [np.mean([r["symbols"] for r in by(data, scheme, e)]) / 1e3 for e in etas], \
            [np.mean([r["completion_time"] for r in by(data, scheme, e)]) for e in etas]

    etas, xs, ys = front(rows, "gosc")
    ax.plot(xs, ys, **STYLE["gosc"])
    for e, x, y in zip(etas, xs, ys):
        if e in (0.0, 1e-5, 1e-4, 1e-3):
            lab = r"$\eta$=0" if e == 0 else rf"$\eta$={e:.0e}".replace("e-0", "e-")
            off = (-5, 4) if e == 0 else (-4, -9)
            ax.annotate(lab, (x, y), textcoords="offset points", xytext=off, fontsize=5.5,
                        color="#2a78d6", ha="right")
    _, xt, yt = front(tv, "gosc_tv")
    ax.plot(xt, yt, **STYLE["gosc_tv"])
    for s in ["sem", "raw", "nl", "report", "gosc_novoi"]:
        st = dict(STYLE[s])
        r = by(dflt, s)
        ax.plot([np.mean([q["symbols"] for q in r]) / 1e3], [np.mean([q["completion_time"] for q in r])],
                linestyle="none", **st)
    g = np.mean([q["completion_time"] for q in by(dflt, "genie")])
    ax.axhline(g, **{k: v for k, v in STYLE["genie"].items() if k != "marker"}, linewidth=1.0)
    ax.set_xscale("log")
    ax.set_xlabel("Channel uses per mission ($\\times 10^3$, log scale)")
    ax.set_ylabel("Completion time (slots)")
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.2))
    fig.savefig(os.path.join(FIG, "fig_pareto.pdf"))
    plt.close(fig)
    return {"tom": {e: summarize(by(rows, "gosc", e)) for e in etas},
            "tv": {e: summarize(by(tv, "gosc_tv", e)) for e in sorted({r["param"] for r in tv})}}


def traj_plot():
    with open(os.path.join(RES, "trajectories.json")) as f:
        tr = json.load(f)
    fig, axes = plt.subplots(1, 2, figsize=(W1, 1.85), sharey=True)
    agent_colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for ax, s in zip(axes, ["sem", "gosc"]):
        d = tr[s]
        for k, path in enumerate(d["trajectories"]):
            p = np.array(path)
            ax.plot(p[:, 1], p[:, 0], color=agent_colors[k % 6], linewidth=0.9)
        v = np.array([divmod(c, 32) for c in d["victims"]])
        ax.scatter(v[:, 1], v[:, 0], marker="x", color="#0b0b0b", s=14, linewidths=1.0, zorder=5)
        ax.scatter([16], [16], marker="^", color="#0b0b0b", s=18, zorder=5)
        ax.set_xlim(-0.5, 31.5)
        ax.set_ylim(31.5, -0.5)
        ax.set_aspect("equal")
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{NAMES[s].split(' (')[0]}: {d['completion_time']} slots", fontsize=7)
    fig.savefig(os.path.join(FIG, "fig_traj.pdf"))
    plt.close(fig)


def fmt(x, d=1):
    return "--" if x != x else f"{x:.{d}f}"


def paired_diff(rows, a, b, nboot=4000):
    """mean over seeds of T_a - T_b and its 95% bootstrap CI (same seeds)."""
    ta = {r["seed"]: r["completion_time"] for r in by(rows, a)}
    tb = {r["seed"]: r["completion_time"] for r in by(rows, b)}
    d = np.array([ta[k] - tb[k] for k in sorted(ta) if k in tb], dtype=float)
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(nboot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(d.mean()), float(lo), float(hi)


def default_table():
    rows = load("default")
    order = ["report", "raw", "nl", "sem", "gosc", "genie"]
    summ = {s: summarize(by(rows, s)) for s in order}
    for s in order:
        summ[s]["diff_vs_gosc"] = None if s == "gosc" else paired_diff(rows, s, "gosc")
    lines = []
    for s in order:
        m = summ[s]
        d = m["diff_vs_gosc"]
        dcell = "--" if d is None else f"{d[0]:+.1f} [{d[1]:+.1f}, {d[2]:+.1f}]"
        name = NAMES[s]
        if s == "gosc":
            name = r"\textbf{" + name + "}"
        if s == "genie":
            lines.append(r"\midrule")
        lines.append(f"{name} & {fmt(m['T'])} $\\pm$ {fmt(m['T_ci'])} & {dcell} & "
                     f"{fmt(m['kbits'], 2)} & {fmt(m['ksym'], 2)} & {fmt(m['latency'], 2)} & "
                     f"{fmt(m['false'], 2)} \\\\")
    with open(os.path.join(FIG, "table_default.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return summ


def llm_table():
    """Planner-only vs mixed LLM/planner teams (results/llm_mixed.jsonl), on the seeds
    available for every compared configuration."""
    path = os.path.join(RES, "llm_mixed.jsonl")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        rows = [json.loads(line) for line in f]
    rng = np.random.default_rng(0)
    T = {}
    for r in rows:
        T.setdefault((r["team"], r["scheme"]), {})[r["seed"]] = r

    def boot(d):
        d = np.asarray(d, float)
        b = d[rng.integers(0, len(d), size=(100000, len(d)))].mean(axis=1)
        lo, hi = np.percentile(b, [2.5, 97.5])
        return [float(d.mean()), float(lo), float(hi), int((d > 0).sum()), int(len(d))]

    def seeds(*keys):
        common = None
        for k in keys:
            ks = set(T.get(k, {}))
            common = ks if common is None else common & ks
        return sorted(common or [])

    out, lines = {}, []
    teams = [("planner", "Planner only"), ("mixed", "Mixed, Qwen2.5-3B"), ("mixed_llama", "Mixed, Llama-3.2-3B")]
    for team, name in teams:
        if (team, "gosc") not in T:
            continue
        ss = seeds((team, "sem"), (team, "gosc"), (team, "genie"))
        first = True
        for sch in ("sem", "gosc", "genie"):
            r = [T[(team, sch)][k] for k in ss]
            m = summarize(r)
            if team != "planner":
                d = sum(x["llm_decisions"] for x in r)
                m["llm_decisions_per_mission"] = d / len(r)
                m["llm_agree"] = sum(x["llm_agree"] for x in r) / d
                m["llm_fallbacks"] = sum(x["llm_fallbacks"] for x in r)
            out[f"{team}/{sch}"] = m
            label = f"{name} ({len(ss)})" if first else ""
            first = False
            sname = {"sem": "Semantic, periodic", "gosc": r"\textbf{GOSC}", "genie": "Genie"}[sch]
            ch = "--" if sch == "genie" else fmt(m["ksym"], 1)
            lines.append(f"{label} & {sname} & {fmt(m['T'])} $\\pm$ {fmt(m['T_ci'])} & {ch} \\\\")
        lines.append(r"\midrule")
        out[f"{team}/sem_minus_gosc"] = boot([T[(team, "sem")][k]["completion_time"]
                                              - T[(team, "gosc")][k]["completion_time"] for k in ss])
        if team != "planner":
            for sch in ("sem", "gosc", "genie"):
                cs = seeds((team, sch), ("planner", sch))
                out[f"{team}_minus_planner/{sch}"] = boot(
                    [T[(team, sch)][k]["completion_time"] - T[("planner", sch)][k]["completion_time"]
                     for k in cs])
            cs = seeds((team, "sem"), (team, "gosc"), ("planner", "sem"), ("planner", "gosc"))
            # difference in degradation: (mixed - planner | sem) - (mixed - planner | gosc)
            out[f"{team}/degradation_difference"] = boot(
                [(T[(team, "sem")][k]["completion_time"] - T[("planner", "sem")][k]["completion_time"])
                 - (T[(team, "gosc")][k]["completion_time"] - T[("planner", "gosc")][k]["completion_time"])
                 for k in cs])
    for team in ("planner", "mixed"):
        if (team, "gosc_bits") in T:
            cs = seeds((team, "gosc_bits"), (team, "gosc"))
            out[f"{team}/bits_minus_gosc"] = boot([T[(team, "gosc_bits")][k]["completion_time"]
                                                   - T[(team, "gosc")][k]["completion_time"] for k in cs])
            out[f"{team}/bits_ksym"] = float(np.mean([T[(team, "gosc_bits")][k]["symbols"] for k in cs]) / 1e3)
            out[f"{team}/gosc_ksym_on_bits_seeds"] = float(np.mean([T[(team, "gosc")][k]["symbols"] for k in cs]) / 1e3)
    if lines and lines[-1] == r"\midrule":
        lines.pop()
    with open(os.path.join(FIG, "table_llm.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return out


def main():
    os.makedirs(FIG, exist_ok=True)
    summary = {"default": default_table()}
    curve_plot()
    summary["snr"] = sweep_plot("snr", "Reference SNR at 100 m (dB)", "fig_snr.pdf")
    sweep_bits_plot("snr", "Reference SNR at 100 m (dB)", "fig_snr_sym.pdf")
    sweep_pair_plot("snr", "Reference SNR at 100 m (dB)", "fig_snr_pair.pdf")
    summary["budget"] = sweep_plot("budget", "Channel uses per agent per slot, $W$", "fig_budget.pdf",
                                   xlog=True, xticks=[25, 50, 100, 200, 400])
    summary["agents"] = sweep_plot("agents", "Number of agents, $K$", "fig_agents.pdf",
                                   xticks=[2, 4, 6, 8, 10])
    summary["pareto"] = pareto_plot()
    summary["llm"] = llm_table()
    traj_plot()
    with open(os.path.join(RES, "summary.json"), "w") as f:
        json.dump(summary, f, indent=1, default=float)
    print("figures written to", os.path.abspath(FIG))


if __name__ == "__main__":
    main()
