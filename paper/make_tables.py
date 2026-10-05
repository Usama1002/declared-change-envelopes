"""LaTeX tables of the paper, generated from the result summaries in results/ (summary.json and the files written by the
analysis scripts). Output: paper/tables/*.tex."""
import json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from config import RESULTS as RES

S = json.load(open(f"{RES}/summary.json"))
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tables")
os.makedirs(OUT, exist_ok=True)
UPD = ["med", "legal", "policy", "edit"]
MN = {"qwen1.5b": "Qwen2.5-1.5B", "qwen7b": "Qwen2.5-7B", "llama8b": "Llama-3.1-8B", "qwen32b": "Qwen2.5-32B", "mistral7b": "Mistral-7B", "qwen72b": "Qwen2.5-72B"}
pct = lambda x: f"{100 * x:.1f}"


def sg(x):
    """Signed percentage with a math minus."""
    return f"{100 * x:+.1f}".replace("-", "$-$")


def t_footprint():
    L = [r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Model & Update & $m(E)$ & effect & in-scope $\Delta$ & $\lambda$ (s0/s1) & $\lambda_\tau$ & $\kappa$ & net acc.\ / churn out \\",
         r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        first = True
        for u in UPD:
            e = S[m]["updates"].get(u)
            if not e:
                continue
            s = e["seed0"]
            L.append(f"{MN[m] if first else ''} & {u} & {pct(s['envelope_mass'])} & {sg(s['in_effect'])} & {pct(s['in_change_rate'])} & "
                     f"{pct(s['leakage'])}/{pct(e['seed1']['leakage']) if 'seed1' in e else '--'} & {pct(s['leakage_decisive'])} & {pct(s['containment'])} & "
                     f"{sg(s['out_acc_change'])} / {pct(s['out_neg_flip'] + s['out_pos_flip'])} \\\\")
            first = False
        n = S[m]["null"]
        L.append(f" & \\emph{{null}} & -- & -- & -- & {pct(n['flip_rate'])} & {pct(n['flip_rate_decisive'])} & -- & -- \\\\")
        if m == "qwen7b":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/footprint.tex", "w").write("\n".join(L))


def t_decl():
    cols = [(m, u) for m in ["qwen7b", "llama8b"] for u in UPD]
    L = [r"\begin{tabular}{l" + "r" * len(cols) + "}", r"\toprule",
         " & " + " & ".join([f"\\multicolumn{{4}}{{c}}{{{MN[m]}}}" for m in ["qwen7b", "llama8b"]]) + r" \\",
         r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}",
         "Declaration & " + " & ".join(u for _, u in cols) + r" \\", r"\midrule"]
    rows = [("nl_judge", "NL judge"), ("knn_data", "data kNN"), ("fragility", "fragility"), ("fragility+judge", "fragility + judge"),
            ("pilot_forecast", "pilot forecast"), ("posthoc_learned", "post-hoc (reference)")]
    for k, name in rows:
        row = []
        for m, u in cols:
            d = S[m]["updates"].get(u, {}).get("declarations", {}).get(k)
            row.append(f"{d['auroc_out']:.2f}" if d and d["auroc_out"] == d["auroc_out"] else "--")
        L.append(f"{name} & " + " & ".join(row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/declarations.tex", "w").write("\n".join(L))


SCOPE = {"pub_HuatuoGPT-o1-7B": ("medical reasoning", "medical"), "pub_HuatuoGPT-o1-8B": ("medical reasoning", "medical"),
         "pub_OpenThinker-7B": ("math/code/science reasoning", "stem"), "pub_Qwen2.5-7B-CLIPPER": ("claim verification in books", None),
         "pub_Llama-3.1-SuperNova-Lite": ("general distillation", None), "pub_ToolACE-8B": ("function calling", None),
         "pub_Llama-3.1-8b-ITA": ("Italian language", None), "pub_Vikhr-Llama3.1-8B-Instruct-R-21-09-24": ("Russian language", None),
         "pub_HallOumi-8B": ("hallucination detection", None), "pub_Qwen2.5-7B-Instruct-Uncensored": ("removing refusals", None)}


def t_public():
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Fine-tune (base) & Declared purpose & changed & changed$_\tau$ & net acc. & neg.\ / pos.\ flips \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        for t, v in S[m]["public"].items():
            sc = SCOPE.get(t, ("?", None))[0]
            nm = t[4:].replace("_", r"\_")
            L.append(f"{nm} ({'Q' if m == 'qwen7b' else 'L'}) & {sc} & {pct(v['flip'])} & {pct(v['flip_decisive'])} & "
                     f"{sg(v['acc_new'] - v['acc_base'])} & {pct(v['neg_flip'])} / {pct(v['pos_flip'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/public.tex", "w").write("\n".join(L))


def t_anchor_main():
    """Main-text table: one row per (model, update); leakage with bootstrap 95% CI; best certificate with the mixture
    sequence (median samples over 20 audit orders) and the one-sided test at the same declared eps."""
    X = json.load(open(f"{RES}/extra.json"))
    fm = lambda x: f"{int(x):,}".replace(",", "{,}")
    L = [r"\begin{tabular}{llrlrlll}", r"\toprule",
         r" & & \multicolumn{2}{c}{LoRA} & \multicolumn{2}{c}{anchored ($\beta{=}1$)} & \multicolumn{2}{c}{best certificate at $\alpha/4$} \\",
         r"\cmidrule(lr){3-4}\cmidrule(lr){5-6}\cmidrule(lr){7-8}",
         r"Model & Update & effect & $\lambda$ [95\% CI] & effect & $\lambda$ [95\% CI] & $\varepsilon$ ($\beta$) & samples: CS / one-sided \\",
         r"\midrule"]
    ci = lambda m, tag: X["ci"].get(f"{m}/{tag}", {}).get("leakage")
    fci = lambda c: f"[{100 * c[1]:.1f}, {100 * c[2]:.1f}]" if c else ""
    for m in ["qwen7b", "llama8b"]:
        V = S[m]["variants"]
        for u, bt, kt, lab in [("med", "med_s0", "med_kl1", "med"), ("legal", "legal_s0", "legal_kl1", "legal"), ("policy", "policy_s0", "policy_kl1", "policy"),
                               ("edit", "edit_mix", "edit_mixkl1", r"edit$^\dagger$")]:
            b = V.get(f"{u}_base") if bt.endswith("_s0") else V.get(bt)
            k = V.get(kt)
            if not b or not k:
                continue
            best = None
            cands = [f"{u}_kl10", f"{u}_kl1", f"{u}_kl0.1"] if u != "edit" else ["edit_mixkl1"]
            for tag in cands:
                v = V.get(tag)
                if not v:
                    continue
                for e in ["0.005", "0.01", "0.02", "0.05"]:
                    pc = v["certificate"].get(f"perm_cond_eps{e}")
                    if pc and pc["frac_certified"] >= 0.5:
                        po = v["certificate"].get(f"perm_os_eps{e}", {})
                        cand = (float(e), tag, pc["median_n"], po.get("median_n") if po.get("frac_certified", 0) >= 0.5 else None)
                        if best is None or cand[0] < best[0]:
                            best = cand
                        break
            if best is None:
                bs = "no & --"
            else:
                beta = best[1].split("kl")[-1]
                bs = f"{best[0] * 100:g}\\% ({beta}) & {fm(best[2])} / {fm(best[3]) if best[3] else '--'}"
            L.append(f"{MN[m]} & {lab} & {sg(b['in_effect'])} & {pct(b['leakage'])} {fci(ci(m, bt))} & {sg(k['in_effect'])} & {pct(k['leakage'])} {fci(ci(m, kt))} & {bs} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/anchor_main.tex", "w").write("\n".join(L))


def t_edit():
    """Edit update: LoRA recipes vs locality-preserving editors, with the neighborhood stratum separately."""
    X = json.load(open(f"{RES}/extra.json"))["edit_compare"]
    names = [("edit_s0", "LoRA, counterfactual only"), ("edit_kl1", r"\quad + anchoring ($\beta{=}1$)"),
             ("edit_mix", "LoRA + 400 true facts"), ("edit_mixkl1", r"\quad + anchoring ($\beta{=}1$)"),
             ("edit_memit", "MEMIT"), ("edit_memitw60k", r"MEMIT, covariance weight $6\cdot10^4$"), ("edit_alphaedit", "AlphaEdit")]
    L = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Model & Method & efficacy & $\lambda$ & $\lambda$ excl.\ nbrs. & nbrs.\ changed & o.o.s.\ acc. & ES / PS / NS \\", r"\midrule"]
    def cfm(m, tag):
        p = f"{RES}/{m}/{tag}_cfmetrics.json"
        if not os.path.exists(p):
            return "--"
        d = json.load(open(p))["post"]
        return f"{100 * d['ES']:.0f} / {100 * d['PS']:.0f} / {100 * d['NS']:.0f}"
    for m in ["qwen7b", "llama8b"]:
        for tag, nm in names:
            v = X.get(f"{m}/{tag}")
            if not v:
                continue
            L.append(f"{MN[m]} & {nm} & {sg(v['effect'])} & {pct(v['leak'])} & {pct(v['leak_excl_neighbors'])} & {pct(v['neighbor_flip'])} & {sg(v['oos_acc_change'])} & {cfm(m, tag)} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(f"{OUT}/edit.tex", "w").write("\n".join(L))


def t_controls():
    """Appendix: every alternative way to reduce leakage, med and policy, Qwen and Llama (development audit set)."""
    Cf = json.load(open(f"{RES}/confirm.json"))
    X = json.load(open(f"{RES}/extra.json"))
    L = [r"\begin{tabular}{lllrr}", r"\toprule", r"Model & Update & Method & effect & $\lambda$ \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        V = S[m]["variants"]
        for u in ["med", "policy"]:
            rows = [("plain LoRA (default)", V.get(f"{u}_base"))]
            sw = [dict(leakage=v["leak"], in_effect=v["effect"], k=k) for k, v in Cf[m]["sweep"].items() if k.startswith(f"sweep_{u}_")]
            if V.get(f"{u}_lr3e-5"):
                sw.append(dict(leakage=V[f"{u}_lr3e-5"]["leakage"], in_effect=V[f"{u}_lr3e-5"]["in_effect"], k=f"sweep_{u}_lr3e-5_e2"))
            if sw:
                lo = min(sw, key=lambda d: d["leakage"])
                _, _, lr, ep = lo["k"].split("_")
                rows.append((f"plain LoRA, lowest-leakage of {len(sw) + 1} lr$\\times$epoch configs (lr {lr[2:]}, {ep[1:]} ep.)", lo))
            for tag, nm in [(f"{u}_scale0.25", "adapter scaled by 0.25"), (f"{u}_scale0.5", "adapter scaled by 0.5"),
                            (f"{u}_l2d1", "L2 penalty to incumbent ($\\mu{=}1$)"), (f"{u}_replay", "incumbent replay (1{,}000)"),
                            (f"{u}_genkl1", "KL on generic chat text ($\\beta{=}1$)"), (f"{u}_mcgenkl1", "KL on generic multiple choice ($\\beta{=}1$)"),
                            (f"{u}_klall", "KL on all traffic incl.\\ envelope ($\\beta{=}1$)"), (f"{u}_klall10", "KL on all traffic incl.\\ envelope ($\\beta{=}10$)"),
                            (f"{u}_kl1", "anchoring on out-of-scope traffic ($\\beta{=}1$)"), (f"{u}_kl10", "anchoring on out-of-scope traffic ($\\beta{=}10$)")]:
                rows.append((nm, V.get(tag)))
            j = X.get("judge_anchor", {}).get(f"{m}/{u}")
            if j:
                rows.append(("anchoring on the NL-judge envelope ($\\beta{=}1$)$^\\ast$", dict(in_effect=j["effect"], leakage=j["leak_vs_judge_env"])))
            first = True
            for nm, v in rows:
                if not v:
                    continue
                L.append(f"{MN[m] if first else ''} & {u if first else ''} & {nm} & {sg(v['in_effect'])} & {pct(v['leakage'])} \\\\")
                first = False
            L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(f"{OUT}/controls.tex", "w").write("\n".join(L))


def t_effect():
    """Appendix: medical effect with adequate power (all 6,277 medical pool queries and MedQA-USMLE test)."""
    L = [r"\begin{tabular}{llrrrr}", r"\toprule",
         r"Model & Variant & gain, pool medical & gain, MedQA & $-$ plain, pool medical & $-$ plain, MedQA \\", r"\midrule"]
    f = lambda v: f"{100 * v[0]:+.1f} [{100 * v[1]:+.1f}, {100 * v[2]:+.1f}]".replace("-", "$-$")
    for m in ["qwen7b", "llama8b"]:
        E = json.load(open(f"{RES}/effect_{m}.json"))
        names = sorted({k.split("/")[0] for k in E["gain"]}, key=lambda x: (not x.startswith("plain"), x))
        for n in names:
            vp = E["vs_plain"].get(f"{n}/pool_medical"); vq = E["vs_plain"].get(f"{n}/medqa")
            L.append(f"{MN[m]} & {n.replace('_', ' ')} & {f(E['gain'][n + '/pool_medical'])} & {f(E['gain'][n + '/medqa'])} & {f(vp) if vp else '--'} & {f(vq) if vq else '--'} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(f"{OUT}/effect.tex", "w").write("\n".join(L))


def t_variants():
    """Appendix: every trained variant with leakage, effect, and permutation-median certificates at alpha/4; split in
    three tables (Qwen-7B, Llama-8B, Mistral-7B + Qwen-32B) so that each fits on a page."""
    CAND = ("_base", "_kl0.1", "_kl1", "_kl10")
    fm = lambda x: f"{int(x):,}".replace(",", "{,}")
    for fname, models in [("variants_qwen", ["qwen7b"]), ("variants_llama", ["llama8b"]), ("variants_other", ["mistral7b", "qwen32b"])]:
        L = [r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Model & Variant & effect & $\lambda$ & $\lambda_\tau$ & $U$ & certified $\lambda$: median samples over 20 orders \\", r"\midrule"]
        for m in models:
            if m not in S:
                continue
            for tag in sorted(S[m]["variants"]):
                v = S[m]["variants"][tag]; c = v["certificate"]
                star = "$^\\ast$" if tag.endswith(CAND) and not tag.startswith("edit") else ""
                cert = "no"
                for e in ["0.005", "0.01", "0.02", "0.05"]:
                    pc = c.get(f"perm_cond_eps{e}")
                    if pc and pc["frac_certified"] >= 0.5:
                        cert = f"{float(e) * 100:g}\\%: {fm(pc['median_n'])} ({fm(pc['q25'])}--{fm(pc['q75'])}); {int(100 * pc['frac_certified'])}\\% of orders"
                        break
                L.append(f"{MN[m]} & {tag.replace('_', r'\_')}{star} & {sg(v['in_effect'])} & {pct(v['leakage'])} & {pct(v['leakage_decisive'])} & {pct(c['ucb_full_cond'])} & {cert} \\\\")
            L.append(r"\midrule")
        L[-1] = r"\bottomrule"
        L.append(r"\end{tabular}")
        open(f"{OUT}/{fname}.tex", "w").write("\n".join(L))


def t_scale():
    """Scale and third family: plain vs anchored (beta=1) for med and policy."""
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r" & & & \multicolumn{2}{c}{LoRA} & \multicolumn{2}{c}{anchored ($\beta{=}1$)} \\", r"\cmidrule(lr){4-5}\cmidrule(lr){6-7}",
         r"Model & Update & floor & effect & $\lambda$ & effect & $\lambda$ \\", r"\midrule"]
    for m in ["qwen7b", "llama8b", "mistral7b", "qwen32b", "qwen72b"]:
        if m not in S:
            continue
        V = S[m]["variants"]
        for u in ["med", "policy"]:
            b, k = V.get(f"{u}_base"), V.get(f"{u}_kl1")
            if not b:
                continue
            L.append(f"{MN[m]} & {u} & {pct(S[m]['null']['flip_rate'])} & {sg(b['in_effect'])} & {pct(b['leakage'])} & "
                     f"{sg(k['in_effect']) if k else '--'} & {pct(k['leakage']) if k else '--'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/scale.tex", "w").write("\n".join(L))


def t_updtypes():
    """Update types (LoRA, full fine-tuning, decision-token GRPO) for med and policy on Qwen-7B and Llama-8B: effect,
    leakage with bootstrap interval, and the smallest epsilon certified (median samples) at level alpha/4."""
    R9 = json.load(open(f"{RES}/update_types.json"))
    rows = [("LoRA", "plain", "{u}_s0"), ("LoRA", r"anchored $\beta{=}1$", "{u}_kl1"), ("LoRA", r"anchored $\beta{=}10$", "{u}_kl10"),
            ("full FT", "plain, lr $10^{-6}$", "{u}_fftlr1e-6"), ("full FT", r"plain, lr $3{\cdot}10^{-6}$", "{u}_fftlr3e-6"), ("full FT", "plain, lr $10^{-5}$", "{u}_fft"),
            ("full FT", r"anchored $\beta{=}10$, lr $10^{-5}$", "{u}_fftkl10"),
            ("full FT", r"anchored $\beta{=}10$, 64 anchors, lr $10^{-6}$", "{u}_fftkl10ab64lr1e-6"),
            ("full FT", r"anchored $\beta{=}10$, 64 anchors, lr $3{\cdot}10^{-6}$", "{u}_fftkl10ab64lr3e-6"),
            ("full FT", r"anchored $\beta{=}10$, 128 anchors, lr $3{\cdot}10^{-6}$", "{u}_fftkl10ab128lr3e-6"),
            ("full FT", r"plain, lr $3{\cdot}10^{-6}$, seed 1", "{u}_fftlr3e-6s1"), ("full FT", r"anchored $\beta{=}10$, 64 anchors, lr $3{\cdot}10^{-6}$, seed 1", "{u}_fftkl10ab64lr3e-6s1"),
            ("GRPO", "reference KL 0.04", "{u}_grpo"), ("GRPO", "reference KL 1", "{u}_grporef1"), ("GRPO", r"ref.\ KL 0.04 + anchored $\beta{=}1$", "{u}_grpokl1"),
            ("GRPO", "reference KL 0.04, seed 1", "{u}_grpos1"), ("GRPO", "reference KL 1, seed 1", "{u}_grporef1s1"), ("GRPO", r"anchored, seed 1", "{u}_grpokl1s1")]
    cert = lambda r: (f"{100 * r['cert']['eps']:g}\\% ({int(r['cert']['n']):,})".replace(",", "{,}") if r.get("cert") else "--")
    L = [r"\begin{tabular}{lllrrlrrl}", r"\toprule",
         r" & & & \multicolumn{3}{c}{Qwen2.5-7B} & \multicolumn{3}{c}{Llama-3.1-8B} \\", r"\cmidrule(lr){4-6}\cmidrule(lr){7-9}",
         r"Update & Type & Variant & effect & $\lambda$ & certified & effect & $\lambda$ & certified \\", r"\midrule"]
    for u in ["med", "policy"]:
        for typ, name, t in rows:
            tag = t.format(u=u); cells = []
            for m in ["qwen7b", "llama8b"]:
                r = R9.get(m, {}).get(u, {}).get(tag)
                cells += [sg(r["in_effect"]), f"{pct(r['leakage'])}", cert(r)] if r else ["--", "--", "--"]
            L.append(f"{u} & {typ} & {name} & " + " & ".join(cells) + r" \\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"; L.append(r"\end{tabular}")
    open(f"{OUT}/updtypes.tex", "w").write("\n".join(L))


def t_rlvr():
    """GSM8K RLVR: in-scope effect (GSM8K test accuracy), multiple-choice leakage, free-form out-of-scope change."""
    R9 = json.load(open(f"{RES}/update_types.json"))
    L = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Model & Variant & train reward & GSM8K acc. $\Delta$ & GSM8K chg. & MC $\lambda$ & NQ chg. (sem.) & MC certified \\", r"\midrule"]
    for m in ["qwen1.5b", "qwen7b", "llama8b"]:
        M = R9.get(m, {}).get("math", {})
        for tag, name, suf in [("math_grpo", "RLVR", "_math"), ("math_grpokl1", "RLVR + anch. (MC)", "_math"), ("math_grpokl1ff", "RLVR + anch. (MC + free-form)", "_math"),
                               ("math_grpostrong", "RLVR strong", "_mathstrong"), ("math_grpostrongkl1", "RLVR strong + anchored", "_mathstrong")]:
            r = M.get(tag)
            if not r:
                continue
            ff = M.get("freeform" + suf, {}); fs = M.get("freeform_sem" + suf, {})
            g = ff.get(tag, {}).get("gsm8k", {}); sem = fs.get(tag, {})
            cert = f"{100 * r['cert']['eps']:g}\\%" if r.get("cert") else "--"
            L.append(f"{MN[m]} & {name} & {r['reward_first10']:.2f}$\\to${r['reward_last10']:.2f} & "
                     f"{sg(g['acc_change']) if g else '--'} & {pct(g['change']) if g else '--'} & {pct(r['leakage'])} & "
                     f"{pct(sem['nq_semantic_change']) if sem else '--'} & {cert} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/rlvr.tex", "w").write("\n".join(L))


def t_gencert():
    """Certificates on written answers: held-out confirmatory split, 3,000 out-of-scope queries, one pre-registered order."""
    L = [r"\begin{tabular}{llrrrrrl}", r"\toprule",
         r"Model & Update & floor & plain & anchored & anchored (letter) & CS: $n$ to certify 5\% & one-sided: $n$ \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        fl = None
        for u in ["med", "legal", "policy"]:
            p = f"{RES}/gen_check_{m}_oos_{u}_confirm" + ("_floor" if u == "med" else "") + ".json"
            if not os.path.exists(p):
                continue
            d = json.load(open(p))
            fl = d["floor_rebatch_gen"]["flip_rate_gen"] if "floor_rebatch_gen" in d else None
            a, b = d[f"{u}_plainsv"], d[f"{u}_anchsv"]
            cs = b["gen_cert_cs_eps0.05"]; os1 = b["gen_cert_onesided_eps0.05"]
            L.append(f"{MN[m]} & {u} & {pct(fl) if fl is not None else '--'} & {pct(a['flip_rate_gen_conservative'])} & {pct(b['flip_rate_gen_conservative'])} & {pct(b['flip_rate_logit'])} & "
                     f"{(str(cs['n']) if cs['certified'] else 'no (' + pct(cs['ucb']) + ')')} & {os1['n'] if os1['certified'] else 'no'} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/gencert.tex", "w").write("\n".join(L))


def t_confirm_types():
    """Held-out (confirmatory-half) certificates and effects for full FT, GRPO, 1.5B and 72B anchored candidates."""
    R12 = json.load(open(f"{RES}/shift_confirm.json"))["confirm"]
    name = {"fftkl10ab64lr1e-6": r"full FT, anchored ($10^{-6}$)", "grpokl1": "GRPO, anchored",
            "kl10": r"LoRA, anchored ($\beta{=}10$)", "kl1": r"LoRA, anchored ($\beta{=}1$)"}
    L = [r"\begin{tabular}{lllrlr}", r"\toprule",
         r"Model & Update & Candidate & $\lambda$ & certified (samples) & effect [95\% CI] \\", r"\midrule"]
    for key, r in R12.items():
        m, tag = key.split("/"); _, u, var = tag.split("_", 2)
        c = r["cert"]
        cs = (f"{100 * c['eps']:g}\\% ({c['n']:,})".replace(",", "{,}")) if c.get("eps") else f"no ({pct(c['ucb'])})"
        lo, hi = r["effect_ci"]
        L.append(f"{MN[m]} & {u} & {name.get(var, var)} & {pct(r['leak'])} & {cs} & {sg(r['effect'])} [{sg(lo)}, {sg(hi)}] " + r"\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/confirm_types.tex", "w").write("\n".join(L))


def t_chat():
    """Open-ended chat traffic (2,000 OpenAssistant prompts): share of responses whose substance changed."""
    rows = [("floor_rebatch", "incumbent, other batch size (floor)"), ("tiny_adapter_0.001", r"tiny random adapter ($\sigma_B{=}10^{-3}$)"),
            ("med_plainsv", "medical, plain"), ("med_anchsv", r"medical, anchored ($\beta{=}10$)"), ("med_mcffkl1", "medical, anchored + free-form KL"),
            ("legal_plainsv", "legal, plain"), ("legal_anchsv", r"legal, anchored ($\beta{=}1$)"),
            ("policy_plainsv", "policy, plain"), ("policy_anchsv", r"policy, anchored ($\beta{=}10$)"), ("policy_mcffkl1", "policy, anchored + free-form KL")]
    D = {m: json.load(open(f"{RES}/chat_sem_{m}.json")) for m in ["qwen7b", "llama8b"]}
    L = [r"\begin{tabular}{lrrrr}", r"\toprule",
         r" & \multicolumn{2}{c}{Qwen2.5-7B} & \multicolumn{2}{c}{Llama-3.1-8B} \\", r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Variant & string & substance & string & substance \\", r"\midrule"]
    for k, nm in rows:
        c = []
        for m in ["qwen7b", "llama8b"]:
            v = D[m].get(k)
            c += [pct(v["string_change"]), pct(v["semantic_change"])] if v else ["--", "--"]
        L.append(nm + " & " + " & ".join(c) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/chat.tex", "w").write("\n".join(L))


def t_appendix_extra():
    C = json.load(open(f"{RES}/confirm.json"))
    # sweep
    L = [r"\begin{tabular}{llrr}", r"\toprule", r"Update & Configuration & effect & $\lambda$ \\", r"\midrule"]
    for tag, v in sorted(C["qwen7b"]["sweep"].items()):
        _, u, lr, ep = tag.split("_")
        L.append(f"{u} & LoRA lr {lr.replace('lr', '')}, {ep.replace('e', '')} ep. & {sg(v['effect'])} & {pct(v['leak'])} \\\\")
    for u in ["med", "policy"]:
        for tag, name in [(f"{u}_base", "LoRA default (lr 1e-4, 2 ep.)"), (f"{u}_lr3e-5", "LoRA lr 3e-5, 2 ep."), (f"{u}_kl1", r"anchored $\beta{=}1$"), (f"{u}_kl10", r"anchored $\beta{=}10$"), (f"{u}_replay", "incumbent replay")]:
            v = S["qwen7b"]["variants"].get(tag)
            if v:
                L.append(f"{u} & {name} & {sg(v['in_effect'])} & {pct(v['leakage'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/sweep.tex", "w").write("\n".join(L))
    # confirmatory + seeds
    L = [r"\begin{tabular}{llrrrrl}", r"\toprule",
         r"Model & Variant & seeds & $\lambda$ dev (mean $\pm$ sd) & $\lambda$ held-out & effect held-out & certified held-out \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        for key, ent in sorted(C[m]["anchored"].items()):
            sm = ent["summary"]; hs = [x for k, x in ent["seeds"].items() if "leak_conf" in x]
            lc = "/".join(pct(x["leak_conf"]) for x in hs); ec = "/".join(sg(x["effect_conf"]) for x in hs)
            ce = "/".join((f"{x['cert_conf']['eps'] * 100:g}\\%" if x["cert_conf"]["eps"] else "no") for x in hs)
            sd = f"{100 * sm['leak_sd']:.2f}" if sm["leak_sd"] is not None else "--"
            L.append(f"{MN[m]} & {key.replace('_', ' ')} & {sm['n_seeds']} & {pct(sm['leak_mean'])} $\\pm$ {sd} & {lc} & {ec} & {ce} \\\\")
        for key, v in sorted(C[m]["plain_conf"].items()):
            L.append(f"{MN[m]} & {key.replace('_', ' ')} (plain) & 1 & -- & {pct(v['leak_conf'])} & {sg(v['effect_conf'])} & -- \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]
    open(f"{OUT}/confirm.tex", "w").write("\n".join(L))


def t_freeform():
    """Free-form traffic slice: change rates of final answers (GSM8K) and short answers (NQ-open, string and semantic)."""
    names = [("floor_rebatch", "incumbent re-batched (floor)"), ("null", "zero adapter, same batching"), ("med_plain", "medical, LoRA"),
             ("med_anch", r"medical, anchored ($\beta{=}1$)"), ("policy_plain", "policy, LoRA"), ("policy_anch", r"policy, anchored ($\beta{=}1$)")]
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Variant & GSM8K chg. & GSM8K acc. & NQ chg. (string) & NQ chg. (sem.) & NQ acc. \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        R = json.load(open(f"{RES}/freeform_{m}_v2.json"))["results"]
        sp = f"{RES}/freeform_sem_{m}.json"
        Sem = json.load(open(sp)) if os.path.exists(sp) else {}
        tp = f"{RES}/freeform_{m}_tiny.json"
        if os.path.exists(tp):
            T = json.load(open(tp))["results"]
            R = dict(R, **{k: T[k] for k in ("tiny4", "tiny3") if k in T})
        fp_ = f"{RES}/freeform_{m}_ffanch.json"
        if os.path.exists(fp_):
            R = dict(R, **json.load(open(fp_))["results"])
            sp2 = f"{RES}/freeform_sem_{m}_ffanch.json"
            if os.path.exists(sp2):
                Sem = dict(Sem, **json.load(open(sp2)))
        for k, nm in names[:2] + [("tiny4", "random adapter, $\\sigma_B{=}10^{-4}$"), ("tiny3", "random adapter, $\\sigma_B{=}10^{-3}$")] + names[2:] + [("med_mcff", r"medical, anchored + free-form KL"), ("policy_mcff", r"policy, anchored + free-form KL")]:
            v = R.get(k)
            if not v:
                continue
            sem = Sem.get(k, {}).get("nq_semantic_change")
            L.append(f"{MN[m]} & {nm} & {pct(v['gsm8k']['change'])} & {sg(v['gsm8k']['acc_change'])} & {pct(v['nq']['change'])} & {pct(sem) if sem is not None else '--'} & {sg(v['nq']['acc_change'])} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(f"{OUT}/freeform.tex", "w").write("\n".join(L))


def t_safety3():
    """Refusal decisions on 720 unsafe prompts judged by an LLM (XSTest three-way rubric)."""
    p = f"{RES}/safety3.json"
    if not os.path.exists(p):
        return
    D = json.load(open(p))
    L = [r"\begin{tabular}{llrrrrr}", r"\toprule",
         r"Model & Variant & refusal rate & changed & newly complied & newly refused & changed (XSTest / AdvBench) \\", r"\midrule"]
    for m in ["qwen7b", "llama8b"]:
        if m not in D:
            continue
        d = D[m]
        L.append(f"{MN[m]} & incumbent & {pct(d['base_refusal'])} & -- & -- & -- & -- \\\\")
        order = lambda k: (0 if k == "base_rebatch" else 1 if k.startswith("plain") else 2, {"anch_s3": "anch_s0"}.get(k, k) if m == "llama8b" else k)
        for k, v in sorted(d["variants"].items(), key=lambda kv: order(kv[0])):
            nm = {"base_rebatch": "incumbent re-batched (floor)"}.get(k, k.replace("_s", " seed ").replace("anch", "anchored").replace("plain", "LoRA"))
            if m == "llama8b" and k == "anch_s3":
                nm = "anchored seed 0"
            L.append(f" & {nm} & {pct(v['refusal'])} & {pct(v['changed'])} & {pct(v['new_compliance'])} & {pct(v['new_refusal'])} & {pct(v['changed_xstest'])} / {pct(v['changed_advbench'])} \\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L.append(r"\end{tabular}")
    open(f"{OUT}/safety3.tex", "w").write("\n".join(L))


if __name__ == "__main__":
    for fn in [t_footprint, t_decl, t_public, t_anchor_main, t_variants, t_edit, t_controls, t_effect, t_scale, t_updtypes,
               t_rlvr, t_gencert, t_confirm_types, t_chat, t_appendix_extra, t_freeform, t_safety3]:
        fn()
        print("ok", fn.__name__)
