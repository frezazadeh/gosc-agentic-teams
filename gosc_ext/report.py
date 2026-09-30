"""Tables, figures and number macros of the revision-2 experiments.

Reads results_r2/*.json[l] (and the journal's results/ read-only) and writes
figures/{table_*,fig_*}_r2 files and figures/numbers_r2.tex.
Every revision-2 number in the paper comes from here.

    python -m gosc_ext.report
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from experiments.plot import FIG  # noqa: E402  (shared style, read-only)

from .run import ARMS, HEADLINE, OUT  # noqa: E402
from .validate import journal_frozen, load_jsonl, reference  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), "..", "results")
W1, W2 = 3.45, 7.0
NB = 4000
MACROS = {}

NAMES = {"search-W100": "Search", "search-K10": "Search, $K{=}10$", "search-clustered": "Search, clustered",
         "rescue-W100": "Rescue", "rescue-W50": "Rescue, $W{=}50$", "rescue-snr-5": "Rescue, $-5$\\,dB"}
ARM_NAMES = {"exact": "Exact ToM value", "ln0.5": "Log-normal error, $\\sigma{=}0.5$",
             "ln1": "Log-normal error, $\\sigma{=}1$", "ln2": "Log-normal error, $\\sigma{=}2$",
             "quant": "Order of magnitude only", "perm": "Content-independent (permuted)",
             "perm_nofilter": "Permuted, no relevance filter"}
ARM_ORDER = ["exact", "ln0.5", "ln1", "quant", "ln2", "perm", "perm_nofilter"]


def macro(name, value):
    MACROS[name] = value


def load(name):
    with open(os.path.join(OUT, name)) as f:
        return json.load(f)


def boot_ratio(a, g, seed=0):
    """mean(a)/mean(g) with a paired percentile-bootstrap CI over seeds."""
    a, g = np.asarray(a, float), np.asarray(g, float)
    idx = np.random.default_rng(seed).integers(0, len(a), (NB, len(a)))
    r = a[idx].mean(1) / g[idx].mean(1)
    return float(a.mean() / g.mean()), *[float(x) for x in np.percentile(r, [2.5, 97.5])]


def boot_mean(x, seed=0):
    x = np.asarray(x, float)
    idx = np.random.default_rng(seed).integers(0, len(x), (NB, len(x)))
    b = x[idx].mean(1)
    return float(x.mean()), *[float(v) for v in np.percentile(b, [2.5, 97.5])]


def margin(metric, x, target):
    x = np.asarray(x, float)
    return boot_mean((target - x) if metric == "completion_time" else 100 * (x - target))


def fr(v, d=2):
    return f"{v[0]:.{d}f} [{v[1]:.{d}f}, {v[2]:.{d}f}]"


def fs(v, d=1):
    return f"{v[0]:+.{d}f} [{v[1]:+.{d}f}, {v[2]:+.{d}f}]"


def spearman(pairs):
    x, y = np.array(pairs, float).T
    rk = lambda z: np.argsort(np.argsort(z, kind="stable"), kind="stable")
    return float(np.corrcoef(rk(x), rk(y))[0, 1])


# ------------------------------------------------------------------------------
def theory():
    t = load("theory.json")
    a, b = t["W100"], t["W50"]
    macro("SlackHi", f"{100 * a['slack_fraction']:.1f}")
    macro("SlackLo", f"{100 * b['slack_fraction']:.1f}")
    macro("BindExactHi", f"{100 * a['binding_exact_fraction']:.0f}")
    macro("BindExactLo", f"{100 * b['binding_exact_fraction']:.0f}")
    macro("BindGapHi", f"{100 * a['binding_mean_gap']:.1f}")
    macro("BindGapLo", f"{100 * b['binding_mean_gap']:.1f}")
    z = (a["zero_value_fraction"] * a["messages"] + b["zero_value_fraction"] * b["messages"]) / (a["messages"] + b["messages"])
    macro("ZeroValue", f"{100 * z:.0f}")
    w1 = (a["within"]["1.0"] * a["messages"] + b["within"]["1.0"] * b["messages"]) / (a["messages"] + b["messages"])
    w2 = (a["within"]["2.0"] * a["messages"] + b["within"]["2.0"] * b["messages"]) / (a["messages"] + b["messages"])
    macro("WithinOne", f"{100 * w1:.0f}")
    macro("WithinTwo", f"{100 * w2:.0f}")
    na, nb = a["sent_counts"]["CONF"], b["sent_counts"]["CONF"]
    macro("ConfMinRate", f"{100 * (a['conf_rate_min'] * na + b['conf_rate_min'] * nb) / (na + nb):.1f}")
    macro("IntentExcHi", f"{100 * (1 - a['intent_rate_le_eff']):.1f}")
    macro("IntentExcLo", f"{100 * (1 - b['intent_rate_le_eff']):.1f}")


# ------------------------------------------------------------------------------
def noise():
    """Fresh-seed channel-use ratio of the journal's frozen periodic point to each
    value-estimation arm (re-selected per setting), and valuation accuracy."""
    v = load("validate_noise.json")
    jv = journal_frozen()
    rows = load_jsonl("noise")
    rho = {}
    for arm in ARMS:
        pairs = [p for r in rows if r.get("arm") == arm and r["setting"] in HEADLINE
                 for p in r.get("value_pairs", [])]
        rho[arm] = spearman(pairs) if pairs else float("nan")
    rho["exact"] = 1.0
    res = {}
    for st in HEADLINE:
        metric, target = reference(st)
        per = np.array(jv[st]["periodic"]["ksym_per_seed"]) * 1e3
        pm = np.array(jv[st]["periodic"]["metric_per_seed"])
        for arm in ARM_ORDER:
            if arm == "exact" and v.get(f"{st}/exact") is None:
                g = np.array(jv[st]["gosc"]["ksym_per_seed"]) * 1e3
                gm = np.array(jv[st]["gosc"]["metric_per_seed"])
                pt = jv[st]["gosc"]["point"].split("/", 1)[1]
            else:
                e = v.get(f"{st}/{arm}")
                if e is None:
                    res[(st, arm)] = None
                    continue
                g, gm, pt = np.array(e["symbols"], float), np.array(e[metric], float), e["point"]
            res[(st, arm)] = {"ratio": boot_ratio(per, g), "margin": margin(metric, gm, target),
                              "diff": boot_mean((gm - pm) * (1 if metric == "completion_time" else 100)),
                              "ksym": g.mean() / 1e3, "point": pt,
                              "meets": bool(gm.mean() <= target if metric == "completion_time" else gm.mean() >= target)}
    # table
    lines = []
    for arm in ARM_ORDER:
        cells = []
        for st in HEADLINE:
            e = res[(st, arm)]
            if e is None:
                cells.append("--")
                continue
            c = f"{e['ratio'][0]:.2f}\\,{{\\scriptsize[{e['ratio'][1]:.2f}, {e['ratio'][2]:.2f}]}}"
            if not e["meets"]:
                c += "$^\\dagger$"
            cells.append(c)
        r = rho[arm]
        lines.append(f"{ARM_NAMES[arm]} & {r:.2f} & " + " & ".join(cells) + " \\\\")
        if arm == "exact":
            lines.append("\\midrule")
    with open(os.path.join(FIG, "table_noise_r2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # figure
    fig, ax = plt.subplots(figsize=(W1, 2.9))
    cols = ["#2a78d6", "#1baf7a", "#eda100", "#e34948", "#4a3aa7", "#eb6834"]
    mks = ["o", "s", "D", "^", "v", "P"]
    xs = np.arange(len(ARM_ORDER))
    for st, col, mk in zip(HEADLINE, cols, mks):
        ys = [res[(st, a)]["ratio"][0] if res[(st, a)] else np.nan for a in ARM_ORDER]
        ok = [res[(st, a)]["meets"] if res[(st, a)] else False for a in ARM_ORDER]
        ax.plot(xs, ys, color=col, linewidth=1.0, zorder=2, marker=mk, markersize=4.2,
                label=NAMES[st].replace("$", "").replace("{=}", "=").replace("\\,", " "))
        for x, y, m in zip(xs, ys, ok):
            ax.plot([x], [y], marker=mk, markersize=4.2, color=col, zorder=3,
                    markerfacecolor=col if m else "white", markeredgewidth=1.0)
    ax.axhline(1.0, color="#7a7974", linestyle="--", linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xticks(xs)
    ticks = ["exact", "$\\sigma$=0.5", "$\\sigma$=1", "$10^{k}$", "$\\sigma$=2", "perm.", "perm.,\nno filt."]
    ax.set_xticklabels([f"{t}\n$\\rho$={rho[a]:.2f}" for t, a in zip(ticks, ARM_ORDER)], fontsize=6)
    ax.set_ylabel("Channel uses, periodic / GOSC")
    ax.set_xlabel("Value estimate (accuracy decreases to the right)")
    ax.legend(frameon=False, fontsize=5.8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22))
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "fig_noise_r2.pdf"))
    plt.close(fig)
    return res, rho


# ------------------------------------------------------------------------------
def overhead():
    v = load("validate_overhead.json")
    res = {}
    for st in HEADLINE:
        for H in (0, 16, 32, 64):
            e = v[f"{st}/H{H}"]
            metric, target = e["metric"], e["target"]
            if e.get("gosc") and not e.get("periodic"):
                gm = np.array(e["gosc"][metric], float)
                res[(st, H)] = {"only_gosc": True, "margin_g": margin(metric, gm, target),
                                "g_meets": bool(gm.mean() <= target if metric == "completion_time" else gm.mean() >= target)}
                continue
            if not e.get("gosc") or not e.get("periodic"):
                res[(st, H)] = None
                continue
            g, p = e["gosc"], e["periodic"]
            ul = boot_ratio(p["symbols"], g["symbols"])
            tot = boot_ratio(p["total_symbols"], g["total_symbols"])
            gm, pm = np.array(g[metric], float), np.array(p[metric], float)
            res[(st, H)] = {"ul": ul, "tot": tot,
                            "g_meets": bool(gm.mean() <= target if metric == "completion_time" else gm.mean() >= target),
                            "p_meets": bool(pm.mean() <= target if metric == "completion_time" else pm.mean() >= target),
                            "margin_g": margin(metric, gm, target), "margin_p": margin(metric, pm, target),
                            "diff": boot_mean((gm - pm) * (1 if metric == "completion_time" else 100)),
                            "g_point": g["point"], "p_point": f"{p['scheme']}/{p['point']}",
                            "g_ksym": np.mean(g["symbols"]) / 1e3, "p_ksym": np.mean(p["symbols"]) / 1e3,
                            "g_tot": np.mean(g["total_symbols"]) / 1e3, "p_tot": np.mean(p["total_symbols"]) / 1e3}
    lines = []
    for block, key in (("Uplink", "ul"), ("Uplink + downlink", "tot")):
        lines.append(f"\\multicolumn{{5}}{{@{{}}l}}{{\\emph{{{block} channel uses, periodic / GOSC}}}}\\\\")
        for st in HEADLINE:
            cells = []
            for H in (0, 16, 32, 64):
                e = res[(st, H)]
                if e is None:
                    cells.append("--")
                    continue
                if e.get("only_gosc"):
                    cells.append("only GOSC" + ("" if e["g_meets"] else "$^\\dagger$"))
                    continue
                c = f"{e[key][0]:.2f}\\,{{\\scriptsize[{e[key][1]:.2f}, {e[key][2]:.2f}]}}"
                if not e["g_meets"]:
                    c += "$^\\dagger$"
                if not e["p_meets"]:
                    c += "$^\\ddagger$"
                cells.append(c)
            lines.append(f"\\quad {NAMES[st]} & " + " & ".join(cells) + " \\\\")
        if key == "ul":
            lines.append("\\midrule")
    with open(os.path.join(FIG, "table_overhead_r2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # cost breakdown, search default, fresh seeds
    lines = []
    lab = {"report": "Report only", "raw": "Raw sensor data", "nl": "Natural language",
           "sem": "Semantic periodic"}
    for H in (0, 32):
        lines.append(f"\\multicolumn{{7}}{{@{{}}l}}{{\\emph{{Overhead $H={H}$ bits}}}}\\\\")
        ent = [(lab[s], v[f"conv/{s}/H{H}"]) for s in ("report", "raw", "nl", "sem")]
        e = v[f"search-W100/H{H}"]
        pname = {"sem_ra": "rate-aware", "sem_raf": "rate-aware + filter", "sem_ras": "suppressing"}[e["periodic"]["scheme"]]
        ent.append((f"Best periodic ({pname})", e["periodic"]))
        ent.append(("\\textbf{GOSC}", e["gosc"]))
        for name, d in ent:
            T = np.mean(d["completion_time"])
            if name.startswith("\\textbf") and T > e["target"]:
                name += "$^\\dagger$"
            ul = np.mean(d["symbols"]) / 1e3
            oh = np.mean(d["ul_overhead_bits"]) / 1e3
            dlb = np.mean(d["dl_bcast_symbols"]) / 1e3
            dlc = np.mean(d["dl_ctrl_symbols"]) / 1e3
            tot = np.mean(d["total_symbols"]) / 1e3
            lines.append(f"\\quad {name} & {T:.1f} & {ul:.1f} & {oh:.1f} & {dlb:.1f} & {dlc:.1f} & {tot:.1f} \\\\")
            if name.startswith("\\textbf") and H == 0:
                lines.append("\\midrule")
    with open(os.path.join(FIG, "table_cost_r2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    return res, v


# ------------------------------------------------------------------------------
SHARED_NAMES = {"search-B600": "Search, $B{=}600$", "search-B300": "Search, $B{=}300$",
                "search-B150": "Search, $B{=}150$", "rescue-B300": "Rescue, $B{=}300$"}


def shared():
    v = load("validate_shared.json")
    rows = load_jsonl("shared")
    res, lines = {}, []
    for st, nm in SHARED_NAMES.items():
        e = v[st]
        metric, target = e["metric"], e["target"]
        sc = 1.0 if metric == "completion_time" else 100.0
        if not e.get("gosc") or not e.get("periodic"):
            lines.append(f"{nm} & {sc * target:.1f} & \\multicolumn{{8}}{{c}}{{target not reached}} \\\\")
            res[st] = None
            continue
        g, p = e["gosc"], e["periodic"]
        gul = np.array(g["symbols"], float) + np.array(g["ul_ctrl_symbols"], float)
        pul = np.array(p["symbols"], float) + np.array(p["ul_ctrl_symbols"], float)
        gm, pm = np.array(g[metric], float), np.array(p[metric], float)
        r = {"ul": boot_ratio(pul, gul), "tot": boot_ratio(p["total_symbols"], g["total_symbols"]),
             "mg": margin(metric, gm, target), "mp": margin(metric, pm, target),
             "diff": boot_mean((gm - pm) * sc), "g_point": g["point"], "p_point": f"{p['scheme']}/{p['point']}",
             "gul": gul.mean() / 1e3, "pul": pul.mean() / 1e3}
        res[st] = r
        lines.append(f"{nm} & {sc * target:.1f} & {sc * gm.mean():.1f} & {fs(r['mg'])} & {r['gul']:.1f} & "
                     f"{sc * pm.mean():.1f} & {fs(r['mp'])} & {r['pul']:.1f} & {fs(r['diff'])} & "
                     f"{fr(r['ul'])} & {fr(r['tot'])} \\\\")
        if st == "search-B150":
            lines.append("\\midrule")
    with open(os.path.join(FIG, "table_shared_r2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # figure: search frontiers under shared budgets (uplink data + control)
    fig, axes = plt.subplots(1, 3, figsize=(W2, 2.3), sharey=True)
    for ax, st in zip(axes, ("search-B600", "search-B300", "search-B150")):
        tab = {}
        for r in rows:
            if r["setting"] == st:
                tab.setdefault((r["family"], r["scheme"], r["point"]), []).append(r)
        def xy(rs):
            return (np.mean([r["symbols"] + r["ul_ctrl_symbols"] for r in rs]) / 1e3,
                    np.mean([r["completion_time"] for r in rs]))
        g = sorted(xy(rs) for (f, s, p), rs in tab.items() if f == "gosc" and len(rs) >= 100)
        ax.plot([a for a, _ in g], [b for _, b in g], color="#2a78d6", marker="o", markersize=3,
                label="GOSC, price-coordinated (each $\\eta$)")
        per = sorted(xy(rs) for (f, s, p), rs in tab.items() if f == "periodic" and len(rs) >= 100)
        env, best = [], None
        for a, b in per:
            if best is None or b < best:
                best = b
                env.append((a, b))
        ax.plot([a for a, _ in per], [b for _, b in per], linestyle="none", marker=".", color="#eb6834",
                alpha=0.35, markersize=3)
        ax.plot([a for a, _ in env], [b for _, b in env], color="#eb6834", marker="s", markersize=3,
                label="Periodic family, best envelope (fair grants)")
        sem = [xy(rs) for (f, s, p), rs in tab.items() if f == "conventional" and len(rs) >= 100]
        if sem:
            ax.plot([sem[0][0]], [sem[0][1]], marker="D", color="#eda100", linestyle="none",
                    label="Semantic periodic, every slot")
        metric, target = reference(st)
        ax.axhline(target, color="#7a7974", linestyle=":", linewidth=0.8, label="Target $1.25\\,T_{\\mathrm{g}}$")
        ax.set_xscale("log")
        ax.set_xticks([1, 2, 5, 10, 20, 50])
        ax.set_xticklabels(["1", "2", "5", "10", "20", "50"])
        ax.minorticks_off()
        ax.set_title(f"Shared budget $B={st.split('B')[1]}$")
        ax.set_xlabel("Uplink channel uses ($\\times10^3$)")
    axes[0].set_ylabel("Completion time (slots)")
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=6.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(FIG, "fig_shared_r2.pdf"))
    plt.close(fig)
    shared_column(rows)
    return res


def shared_column(rows):
    """Single-column version of the shared-budget frontiers (main paper)."""
    fig, axes = plt.subplots(1, 3, figsize=(W1, 1.85), sharey=True)
    for ax, st in zip(axes, ("search-B600", "search-B300", "search-B150")):
        tab = {}
        for r in rows:
            if r["setting"] == st:
                tab.setdefault((r["family"], r["scheme"], r["point"]), []).append(r)

        def xy(rs):
            return (np.mean([r["symbols"] + r["ul_ctrl_symbols"] for r in rs]) / 1e3,
                    np.mean([r["completion_time"] for r in rs]))
        g = sorted(xy(rs) for (f, s, p), rs in tab.items() if f == "gosc" and len(rs) >= 100)
        per = sorted(xy(rs) for (f, s, p), rs in tab.items() if f == "periodic" and len(rs) >= 100)
        env, best = [], None
        for a, b in per:
            if best is None or b < best:
                best = b
                env.append((a, b))
        ax.plot([a for a, _ in g], [b for _, b in g], color="#2a78d6", marker="o", markersize=2.5,
                linewidth=1.0, label="GOSC ($\\eta$ swept)")
        ax.plot([a for a, _ in env], [b for _, b in env], color="#eb6834", marker="s", markersize=2.5,
                linewidth=1.0, label="Periodic family, envelope")
        ax.axhline(reference(st)[1], color="#7a7974", linestyle=":", linewidth=0.8,
                   label="Target $1.25\\,T_{\\mathrm{g}}$")
        ax.set_xscale("log")
        ax.set_xticks([1, 3, 10, 30])
        ax.set_xticklabels(["1", "3", "10", "30"], fontsize=6)
        ax.minorticks_off()
        ax.tick_params(axis="y", labelsize=6)
        ax.set_title(f"$B={st.split('B')[1]}$", fontsize=7)
    axes[1].set_xlabel("Uplink channel uses ($\\times10^3$)", fontsize=7)
    axes[0].set_ylabel("Completion time (slots)", fontsize=7)
    h, lb = axes[0].get_legend_handles_labels()
    fig.legend(h, lb, frameon=False, fontsize=5.8, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 0.0))
    fig.tight_layout(rect=(0, 0.1, 1, 1), w_pad=0.4)
    fig.savefig(os.path.join(FIG, "fig_shared_col_r2.pdf"))
    plt.close(fig)


# ------------------------------------------------------------------------------
def llm():
    rows = [json.loads(line) for line in open(os.path.join(RES, "llm_mixed.jsonl"))]
    new = []
    p = os.path.join(OUT, "llm_r2.jsonl")
    if os.path.exists(p):
        new = [json.loads(line) for line in open(p)]
    seen = {(r["team"], r["scheme"], r["seed"]) for r in rows}
    rows += [r for r in new if (r["team"], r["scheme"], r["seed"]) not in seen]
    teams = [("planner", "Planner only"), ("mixed", "Mixed, Qwen2.5-3B"),
             ("mixed_llama", "Mixed, Llama-3.2-3B"), ("mixed_qwen7b", "Mixed, Qwen2.5-7B")]
    out, lines = {}, []
    for team, nm in teams:
        by = {}
        for r in rows:
            if r["team"] == team:
                by.setdefault(r["scheme"], {})[r["seed"]] = r
        seeds = sorted(set(by.get("sem", {})) & set(by.get("gosc", {})))
        if not seeds:
            continue
        s = np.array([by["sem"][i]["completion_time"] for i in seeds], float)
        g = np.array([by["gosc"][i]["completion_time"] for i in seeds], float)
        ys = np.array([by["sem"][i]["symbols"] for i in seeds], float)
        yg = np.array([by["gosc"][i]["symbols"] for i in seeds], float)
        d = boot_mean(s - g)
        rt = boot_ratio(ys, yg)
        o = {"n": len(seeds), "dT": d, "ratio": rt}
        gen = [by["genie"][i]["completion_time"] for i in seeds if i in by.get("genie", {})]
        agree = [by[sc][i] for sc in ("sem", "gosc") for i in seeds if "llm_decisions" in by[sc][i]]
        if agree:
            o["agree"] = sum(r["llm_agree"] for r in agree) / sum(r["llm_decisions"] for r in agree)
            o["fallbacks"] = sum(r["llm_fallbacks"] for r in agree)
        out[team] = o
        ci = lambda x: (np.mean(x), 1.96 * np.std(x, ddof=1) / np.sqrt(len(x)))
        lines.append(f"{nm} ({len(seeds)}) & Semantic, periodic & {ci(s)[0]:.1f} $\\pm$ {ci(s)[1]:.1f} & "
                     f"{ys.mean() / 1e3:.1f} & \\multirow{{2}}{{*}}{{{fs(d)}}} & \\multirow{{2}}{{*}}{{{rt[0]:.2f}}}\\\\")
        lines.append(f" & \\textbf{{GOSC}} & {ci(g)[0]:.1f} $\\pm$ {ci(g)[1]:.1f} & {yg.mean() / 1e3:.1f} & & \\\\")
        if len(gen) == len(seeds):
            lines.append(f" & Genie & {ci(gen)[0]:.1f} $\\pm$ {ci(gen)[1]:.1f} & -- & & \\\\")
        lines.append("\\midrule")
    lines = lines[:-1]
    with open(os.path.join(FIG, "table_llm_r2.tex"), "w") as f:
        f.write("\n".join(lines) + "\n")
    # interaction with the planner team (same seeds): does periodic degrade more?
    by = {}
    for r in rows:
        by.setdefault((r["team"], r["scheme"]), {})[r["seed"]] = r["completion_time"]
    for team in ("mixed", "mixed_llama", "mixed_qwen7b"):
        if team not in out or (team, "gosc") not in by:
            continue
        seeds = sorted(set(by[(team, "sem")]) & set(by[(team, "gosc")]) & set(by[("planner", "sem")])
                       & set(by[("planner", "gosc")]))
        x = np.array([(by[(team, "sem")][i] - by[("planner", "sem")][i])
                      - (by[(team, "gosc")][i] - by[("planner", "gosc")][i]) for i in seeds], float)
        out[team]["interaction"] = boot_mean(x)
        if (team, "genie") in by and ("planner", "genie") in by:
            sg = sorted(set(by[(team, "genie")]) & set(by[("planner", "genie")]))
            out[team]["genie_gap"] = boot_mean([by[(team, "genie")][i] - by[("planner", "genie")][i] for i in sg])
    mixed = [t for t in ("mixed", "mixed_llama", "mixed_qwen7b") if t in out]
    ag = [out[t]["agree"] for t in mixed]
    macro("LLMAgreeLo", f"{100 * min(ag):.0f}")
    macro("LLMAgreeHi", f"{100 * max(ag):.0f}")
    g = out["mixed"]["genie_gap"]
    macro("LLMGenieGap", f"{g[0]:.1f} [{g[1]:.1f}, {g[2]:.1f}]")
    rr = [out[t]["ratio"][0] for t in mixed]
    lo, hi = f"{min(rr):.1f}", f"{max(rr):.1f}"
    macro("LLMRatioRange", lo if lo == hi else f"{lo}--{hi}")
    f1 = lambda v: f"{v[0]:.1f} [{v[1]:.1f}, {v[2]:.1f}]".replace("[-", "[$-$").replace(", -", ", $-$")
    macro("LLMdTQwen", f1(out["mixed"]["dT"]))
    macro("LLMdTLlama", f1(out["mixed_llama"]["dT"]))
    macro("LLMdTPlanner", f1(out["planner"]["dT"]).replace("-", "$-$", 1) if out["planner"]["dT"][0] < 0 else f1(out["planner"]["dT"]))
    macro("LLMInterQwen", f1(out["mixed"]["interaction"]))
    if "mixed_qwen7b" in out:
        q = out["mixed_qwen7b"]
        sig = q["dT"][1] > 0 or q["dT"][2] < 0
        macro("QSevenText", f"With the stronger Qwen2.5-7B ({q['n']} seeds), the difference was {f1(q['dT'])} slots"
              + (", significant." if sig else ", not significant."))
    else:
        macro("QSevenText", "")
    return out


# ------------------------------------------------------------------------------
def main():
    summary = {}
    theory()
    for name, fn in (("noise", noise), ("overhead", overhead), ("shared", shared), ("llm", llm)):
        try:
            summary[name] = fn()
        except FileNotFoundError as e:
            print("skip", name, e)
    with open(os.path.join(FIG, "numbers_r2.tex"), "w") as f:
        for k, v in MACROS.items():
            f.write(f"\\newcommand{{\\{k}}}{{{v}}}\n")

    def js(o):
        if isinstance(o, dict):
            return {str(k): js(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)):
            return [js(v) for v in o]
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        return o
    with open(os.path.join(OUT, "summary_r2.json"), "w") as f:
        json.dump(js(summary), f, indent=1)
    print(json.dumps(js(summary), indent=1)[:6000])


if __name__ == "__main__":
    main()
