"""Figures of the paper (matplotlib, PDF) from the result summaries in results/. Output: paper/figures/*.pdf."""
import json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import RESULTS as RES

S = json.load(open(f"{RES}/summary.json"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
# high-contrast categorical palette (dark enough on white, distinct hues)
CAT = ["#0057b8", "#d62828", "#1a9e3e", "#e07b00", "#c2185b", "#008b8b", "#6a1b9a", "#5d4037"]
INK, INK2, GRID = "#0b0b0b", "#3a3a3a", "#e3e3e3"
plt.rcParams.update({"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.titlesize": 9.5,
                     "legend.fontsize": 8, "legend.handletextpad": 0.3, "legend.columnspacing": 1.0, "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5, "legend.frameon": False,
                     "font.family": "serif", "pdf.fonttype": 42, "lines.linewidth": 2.0, "lines.markersize": 6})
UPD = ["med", "legal", "policy", "edit"]
UNAME = {"med": "med", "legal": "legal", "policy": "policy", "edit": "edit"}
MNAME = {"qwen7b": "Qwen2.5-7B-Instruct", "llama8b": "Llama-3.1-8B-Instruct"}
DOMS = ["medical", "legal", "news_topic", "entity_facts_edited", "entity_facts_neighbor", "stem", "humanities_social",
        "science_qa", "commonsense", "reading", "sentiment"]
DNAME = {"medical": "medical", "legal": "legal", "news_topic": "news topic", "entity_facts_edited": "edited facts",
         "entity_facts_neighbor": "neighbor facts", "stem": "STEM", "humanities_social": "humanities/soc.",
         "science_qa": "science QA", "commonsense": "commonsense", "reading": "reading", "sentiment": "sentiment"}
SCOPE = {"med": "medical", "legal": "legal", "policy": "news_topic", "edit": "entity_facts_edited"}


def heatmap(which="controlled"):
    rows, labels, scopes = [], [], []
    for m in ["qwen7b", "llama8b"]:
        if m not in S:
            continue
        for u in UPD:
            e = S[m]["updates"].get(u)
            if e and which == "controlled":
                rows.append([e["seed0"]["per_domain_flip"][d] for d in DOMS]); labels.append(f"{MNAME[m].split('-')[0]} {u}")
                scopes.append(SCOPE[u])
        for tag, v in (S[m].get("public", {}).items() if which == "public" else []):
            rows.append([v["per_domain_flip"][d] for d in DOMS]); labels.append(f"{tag[4:]} ({'Q' if m == 'qwen7b' else 'L'})"); scopes.append(None)
    A = np.array(rows) * 100
    cmap = LinearSegmentedColormap.from_list("seq", ["#f7f7f5", "#9ec5f0", "#2a78d6", "#0b3d7a"])
    fig, ax = plt.subplots(figsize=(6.3, 0.17 * len(rows) + 1.05))
    im = ax.imshow(np.clip(A, 0, 60), cmap=cmap, aspect="auto", vmin=0, vmax=60)
    ax.set_xticks(range(len(DOMS))); ax.set_xticklabels([DNAME[d] for d in DOMS], rotation=35, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(labels)
    ax.grid(False)
    for i in range(len(rows)):
        for j in range(len(DOMS)):
            ax.text(j, i, f"{A[i, j]:.0f}", ha="center", va="center", fontsize=6,
                    color="#ffffff" if A[i, j] > 30 else INK)
        if scopes[i]:
            j = DOMS.index(scopes[i])
            ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec=INK, lw=1.2))
    for i in range(1, len(labels)):
        if ("Llama" in labels[i]) != ("Llama" in labels[i - 1]) or (labels[i][:4] in ("Qwen", "Llam")) != (labels[i - 1][:4] in ("Qwen", "Llam")):
            ax.axhline(i - 0.5, color="#ffffff", lw=2.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); cb.set_label("decisions changed (%)")
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_heatmap{'' if which == 'controlled' else '_public'}.pdf"); plt.close(fig)


def margin_law():
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 2.35), sharey=True)
    for ax, m in zip(axes, ["qwen7b", "llama8b"]):
        if m not in S:
            continue
        for k, u in enumerate(UPD):
            e = S[m]["updates"].get(u)
            if not e:
                continue
            ml = e["margin_law"]
            x = [np.sqrt(b["lo"] * b["hi"]) if b["lo"] > 0 else b["hi"] / 2 for b in ml if b["flip"] is not None]
            y = [max(b["flip"], 1e-4) for b in ml if b["flip"] is not None]
            nn = [max(b["n"], 1) for b in ml if b["flip"] is not None]
            se = [1.96 * np.sqrt(max(p, 1e-4) * (1 - p) / n) for p, n in zip(y, nn)]
            ax.fill_between(x, [max(p - e, 1e-4) for p, e in zip(y, se)], [min(p + e, 1) for p, e in zip(y, se)],
                            color=CAT[k], alpha=0.22, lw=0)
            ax.plot(x, y, "-o", ms=5, color=CAT[k], label=u)
        ml = S[m]["updates"]["med"]["margin_law"]
        x = [np.sqrt(b["lo"] * b["hi"]) if b["lo"] > 0 else b["hi"] / 2 for b in ml if b["null"] is not None]
        y = [max(b["null"], 1e-4) for b in ml if b["null"] is not None]
        ax.plot(x, y, "--", color=INK, lw=1.8, label="self-disagreement")
        ax.set_xscale("log"); ax.set_yscale("log"); ax.set_title(MNAME[m])
        ax.set_xlabel(r"incumbent margin $\mu_{f_0}(x)$ (nats)")
    axes[0].set_ylabel("P(change), out of scope")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper center", ncol=5, bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.3, w_pad=0.6); fig.savefig(f"{OUT}/fig_margin.pdf"); plt.close(fig)


DECL = [("nl_judge", "NL judge"), ("knn_data", "data kNN"), ("fragility", "fragility"),
        ("pilot_forecast", "pilot forecast"), ("posthoc_learned", "post-hoc learned")]


def footprint_curves(m="qwen7b"):
    fig, axes = plt.subplots(1, 4, figsize=(6.8, 2.75), sharey=False)
    for ax, u in zip(axes, UPD):
        e = S[m]["updates"].get(u)
        if not e:
            continue
        for k, (key, name) in enumerate(DECL):
            if key in e["declarations"]:
                c = e["declarations"][key]["curve"]
                ax.plot([p["mass"] for p in c], [100 * p["leakage"] for p in c], "-", color=[CAT[0], CAT[1], CAT[2], CAT[3], CAT[6]][k], lw=2.0, label=name)
        for key, sty in [("random", ":"), ("oracle", "--")]:
            c = e["declarations"][key]["curve"]
            ax.plot([p["mass"] for p in c], [100 * p["leakage"] for p in c], sty, color=INK, lw=1.4, label=key)
        t = e["seed0"]
        ax.plot([t["envelope_mass"]], [100 * t["leakage"]], "s", color=INK, ms=8, label="taxonomy envelope", zorder=5)
        ax.axhline(2, color="#777777", lw=1.4, ls="-.", label="2% target")
        ax.set_title(u); ax.set_xlabel("declared mass $m$"); ax.set_xlim(0, 0.92)
        ax.set_ylim(bottom=0)
    axes[0].set_ylabel("leakage $\\lambda$ (%)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0.01, 0.2, 1, 1), pad=0.5, w_pad=0.4); fig.savefig(f"{OUT}/fig_curves_{m}.pdf"); plt.close(fig)


def pareto():
    """Effect vs leakage for every way of trading them off: plain LoRA grid (lr x epochs), adapter-scale interpolation,
    L2 to the incumbent, incumbent replay, KL on generic chat text, and envelope anchoring (declared out-of-scope KL)."""
    Cf = json.load(open(f"{RES}/confirm.json"))
    groups = [("plain LoRA (lr x epochs)", CAT[1], "o"), ("adapter scaling", CAT[3], "v"), ("L2 to incumbent", CAT[4], "D"),
              ("incumbent replay", CAT[6], "P"), ("KL, generic chat text", CAT[2], "X"), ("KL, generic multiple choice", CAT[5], "^"), ("KL, all traffic", "#111111", "*"), ("anchoring, out-of-scope traffic (ours)", CAT[0], "s")]
    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.3))
    for r, m in enumerate(["qwen7b", "llama8b"]):
        V = S[m]["variants"]
        for c, u in enumerate(["med", "policy"]):
            ax = axes[r, c]
            pts = {g[0]: [] for g in groups}
            if f"{u}_base" in V:
                pts[groups[0][0]].append(V[f"{u}_base"])
            for k, v in Cf.get(m, {}).get("sweep", {}).items():
                if k.startswith(f"sweep_{u}_"):
                    pts[groups[0][0]].append(dict(leakage=v["leak"], in_effect=v["effect"]))
            for k, v in V.items():
                if not k.startswith(u + "_"):
                    continue
                lab = k[len(u) + 1:]
                if lab.startswith("scale"):
                    pts[groups[1][0]].append(v)
                elif lab.startswith("l2d"):
                    pts[groups[2][0]].append(v)
                elif lab == "replay":
                    pts[groups[3][0]].append(v)
                elif lab.startswith("genkl"):
                    pts[groups[4][0]].append(v)
                elif lab.startswith("mcgenkl"):
                    pts[groups[5][0]].append(v)
                elif lab.startswith("klall"):
                    pts[groups[6][0]].append(v)
                elif lab in ("kl0.1", "kl1", "kl10"):
                    pts[groups[7][0]].append(v)
            for name, col, mk in groups:
                P = pts[name]
                if not P:
                    continue
                ax.plot([100 * p["leakage"] for p in P], [100 * p["in_effect"] for p in P], mk, ms=12 if mk == "*" else 8, color=col,
                        mec="#ffffff", mew=0.7, label=name, ls="none", alpha=0.95)
            ax.set_xscale("log"); ax.set_xlim(0.9, 45)
            ax.set_xticks([1, 2, 5, 10, 20, 40]); ax.set_xticklabels(["1", "2", "5", "10", "20", "40"]); ax.minorticks_off()
            ax.set_title(f"{MNAME[m].split('-Instruct')[0]}, {u}")
            if r == 1:
                ax.set_xlabel("out-of-scope leakage $\\lambda$ (%)")
            if c == 0:
                ax.set_ylabel("in-scope effect (pp)")
    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.005), markerscale=0.9)
    fig.tight_layout(rect=(0, 0.13, 1, 1), pad=0.3, h_pad=0.5, w_pad=0.6); fig.savefig(f"{OUT}/fig_pareto.pdf"); plt.close(fig)




def utype(tag):
    """Update-type group of a variant tag, for the KL-leakage figure."""
    t = tag.split("_", 1)[1] if "_" in tag else tag
    if tag.startswith("math_"):
        return "RLVR (GSM8K)"
    if "memit" in t or "alphaedit" in t:
        return "knowledge editor"
    if "dpo" in t:
        return "DPO"
    if t.startswith("grpo"):
        return "GRPO, anchored" if "kl" in t else ("GRPO, strong ref. KL" if "ref1" in t else "GRPO")
    if t.startswith("fft"):
        return "full FT, anchored" if "kl" in t else "full FT"
    if t.startswith("kl") or "kl1" in t or "kl10" in t or "kl0.1" in t:
        return "LoRA, anchored"
    return "LoRA, plain"


def kl_leak():
    R = json.load(open(f"{RES}/kl_leak.json"))
    order = [("LoRA, plain", "#8a8a8a", "o"), ("LoRA, anchored", CAT[0], "s"), ("full FT", CAT[1], "^"),
             ("full FT, anchored", CAT[6], "^"), ("GRPO", CAT[3], "D"), ("GRPO, strong ref. KL", CAT[4], "D"),
             ("GRPO, anchored", CAT[2], "D"), ("DPO", CAT[7], "v"), ("knowledge editor", CAT[5], "P"), ("RLVR (GSM8K)", "#111111", "X")]
    fig, ax = plt.subplots(1, 2, figsize=(6.8, 3.15))
    for name, col, mk in order:
        pts = [r for r in R if utype(r["tag"]) == name]
        if not pts:
            continue
        kw = dict(s=34, color=col, marker=mk, alpha=0.9, edgecolors="white", linewidths=0.4)
        ax[0].scatter([r["K"] for r in pts], [100 * r["leak"] for r in pts], label=name, **kw)
        ax[1].scatter([100 * r["pinsker_pointwise"] for r in pts], [100 * r["leak"] for r in pts], **kw)
    for a in ax:
        a.set_xscale("log"); a.set_yscale("log")
    ax[0].set_xlabel("mean out-of-scope letter KL to incumbent (nats)"); ax[0].set_ylabel(r"out-of-scope leakage $\lambda$ (%)")
    lim = [0.6, 100]; ax[1].plot(lim, lim, "--", color=INK, lw=1.3); ax[1].text(1.0, 1.9, "leakage = bound", rotation=33, fontsize=8, color=INK)
    ax[1].set_xlim(0.6, 110); ax[1].set_ylim(0.5, 110)
    ax[1].set_xlabel(r"Lemma 2 bound $P(K \geq g_0^2/2)$ (%)"); ax[1].set_ylabel("leakage (%)")
    ax[0].set_ylim(0.5, 110)
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.01), markerscale=1.2)
    fig.tight_layout(rect=(0, 0.15, 1, 1), pad=0.3, w_pad=1.0); fig.savefig(f"{OUT}/fig_kl_leak.pdf"); plt.close(fig)


if __name__ == "__main__":
    heatmap("public")
    for f in [heatmap, margin_law, footprint_curves, pareto, kl_leak]:
        try:
            f(); print("ok", f.__name__)
        except Exception as ex:
            print("FAIL", f.__name__, repr(ex))
