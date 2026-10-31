# -*- coding: utf-8 -*-
"""Revize makalenin şekillerini ve grafiksel özetlerini runs/revizyon/results/analiz/ altındaki tablolardan yeniden üretir
(Carlito yazı tipi, Türkçe ondalık virgül, Okabe-Ito renkleri, 600 dpi). Önce analiz/revizyon_calistir.py çalıştırılmalıdır.

    python analiz/revizyon_sekilleri.py            # şekiller: makale_sekilleri_revizyon_yeniden/
    python analiz/revizyon_sekilleri.py <klasör>   # başka bir klasöre
"""
import os, sys, json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
O2 = os.path.join(ROOT, "runs", "revizyon", "results", "analiz")
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "makale_sekilleri_revizyon_yeniden"); os.makedirs(OUT, exist_ok=True)
CM = 1 / 2.54; WL = 15 * CM
plt.rcParams.update({"font.family": "Carlito", "font.size": 8.5, "axes.labelsize": 8.5, "xtick.labelsize": 8, "ytick.labelsize": 8,
                     "legend.fontsize": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
                     "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "text.color": "black", "axes.labelcolor": "black", "xtick.color": "black", "ytick.color": "black"})
BLUE, ORANGE, GREEN, VERM, SKY, PURPLE, GREY, YEL = "#0072B2", "#E69F00", "#009E73", "#D55E00", "#56B4E9", "#CC79A7", "#999999", "#F0E442"
def f2(v, d=2):
    return f"{v:.{d}f}".replace(".", ",")
TRD = FuncFormatter(lambda v, p: f2(v, 1) if abs(v * 10 - round(v * 10)) < 1e-9 else f2(v))
PCT = FuncFormatter(lambda v, p: f"%{v * 100:.0f}")

SYN = pd.read_csv(f"{O2}/sentetik_tum_yontemler.csv", encoding="utf-8-sig")
def row(m, c):
    r = SYN[(SYN.yontem == m) & (SYN.kosul == c)]
    assert len(r) == 1, (m, c, len(r))
    return r.iloc[0]
LLM = [("claude-opus-5-5", "Claude Opus 5.5"), ("gemini-3.8-flash", "Gemini 3.8 Flash"), ("claude-sonnet-5-5", "Claude Sonnet 5.5"),
       ("gpt-5-mini", "GPT-5 mini"), ("claude-haiku-5-5", "Claude Haiku 5.5")]

def group_rows(ax, groups, y0=0.0, gap=0.7):
    """groups: [(grup_adı, [satırlar])] -> y konumları (yukarıdan aşağı) ve grup başlıkları."""
    ys, labels, heads = [], [], []
    y = y0
    for g, rows in groups:
        heads.append((g, y)); y -= 0.85
        for r in rows:
            ys.append(y); labels.append(r); y -= 1.0
        y -= gap - 1.0 + 1.0
    return ys, labels, heads

# ------------------------------------------------------------------ Şekil 2: metin türüne göre (makaledeki modeller + ablasyon)
def fig_text_type():
    groups = [("Benzerlik için ince ayarlanmış cümle kodlayıcılar", [("USE", "USE"), ("SBERT-mpnet", "SBERT-mpnet"), ("ALBERT-paraphrase", "ALBERT-paraphrase"),
                                                               ("RoBERTa-güncel", "RoBERTa (all-distilroberta-v1)")]),
              ("Kullanımdan kaldırılmış NLI kodlayıcıları", [("BERT", "BERT (bert-base-nli-mean-tokens)"), ("RoBERTa", "RoBERTa (roberta-base-nli-mean-tokens)")]),
              ("Benzerlik için ince ayarlanmamış", [("BERT-raw", "BERT-raw (bert-base-uncased)"), ("RoBERTa-raw", "RoBERTa-raw (roberta-base)"), ("ALBERT", "ALBERT (albert-base-v2)")]),
              ("Statik kelime vektörleri", [("Word2Vec", "Word2Vec"), ("FastText", "FastText"), ("GloVe", "GloVe")]),
              ("Sözcüksel temel çizgi", [("Temel: TF-IDF", "TF-IDF")])]
    n = sum(len(r) for _, r in groups) + len(groups)
    fig, ax = plt.subplots(figsize=(WL, (n * 0.40 + 1.6) * CM))
    y = 0; yt, yl = [], []
    mk = {"plain": ("o", BLUE, "Düz metin"), "lemma": ("s", ORANGE, "Lemmalanmış"), "stem": ("^", GREEN, "Gövdelenmiş")}
    heads = []
    for g, rows in groups:
        yt.append(y); yl.append(g); heads.append(len(yt) - 1)
        y -= 1
        for m, lab in rows:
            for k, (tt, (mm, col, _)) in enumerate(mk.items()):
                r = SYN[(SYN.yontem == m) & (SYN.kosul == f"en/{tt}/ad")]
                if r.empty:
                    continue
                r = r.iloc[0]; dy = (k - 1) * 0.22
                ax.errorbar(r.hit1, y + dy, xerr=[[r.hit1 - r.hit1_ga_alt], [r.hit1_ga_ust - r.hit1]], fmt=mm, color=col, ms=4.2,
                            mec="white", mew=0.4, elinewidth=0.7, capsize=0, zorder=3)
            yt.append(y); yl.append(lab); y -= 1
        y -= 0.25
    op = row("BDM claude-opus-5-5", "sifir_atis/en").hit1; rnd = row("Temel: Rastgele", "en/plain/ad").hit1
    ax.axvline(op, color=VERM, ls=":", lw=1.0, zorder=1); ax.text(op - 0.01, 0, f"Claude Opus 5.5 ({f2(op)})", color=VERM, ha="right", va="center", fontsize=7.5)
    ax.axvline(rnd, color=GREY, ls="-.", lw=0.8, zorder=1); ax.text(rnd + 0.01, -1.0, f"Rastgele ({f2(rnd)})", color="#555555", ha="left", va="center", fontsize=7.5)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_ylim(y + 0.6, 0.6)
    for k, tl in enumerate(ax.get_yticklabels()):
        if k in heads:
            tl.set_fontstyle("italic"); tl.set_fontweight("bold"); tl.set_color("#333333")
    ax.set_xlim(0, 1.0); ax.xaxis.set_major_formatter(TRD); ax.set_xlabel("Mesaj düzeyi isabet (hit@1), %95 önyükleme GA")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], marker=v[0], color=v[1], ls="", ms=5, label=v[2]) for v in mk.values()]
    ax.legend(handles=hs, frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_hit1.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 4: tüm yöntemler (yalnız ad / ad + tanım)
def best(m, lang, nm, allowed=("plain", "lemma", "stem", "raw")):
    r = SYN[(SYN.yontem == m) & SYN.kosul.str.startswith(lang + "/") & SYN.kosul.str.endswith("/" + nm) &
            SYN.kosul.str.split("/").str[1].isin(allowed)]
    return None if r.empty else r.sort_values("hit1", ascending=False).iloc[0]

def fig_overall():
    TRS = pd.read_csv(f"{O2}/tfidf_turkce_sentetik.csv")
    groups = [
        ("Büyük dil modelleri (sıfır atış, İngilizce metin)", [(f"BDM {m}", lab, "en", "llm") for m, lab in LLM]),
        ("Güncel çok dilli kodlayıcılar (İngilizce metin)", [("Qwen3-Emb-4B", "Qwen3-Embedding-4B", "en", "emb"), ("mE5-large-instruct", "mE5-large-instruct", "en", "emb"),
                                                              ("BGE-M3", "BGE-M3", "en", "emb"), ("Qwen3-Emb-0.6B", "Qwen3-Embedding-0.6B", "en", "emb")]),
        ("Cümle kodlayıcılar (İngilizce metin)", [("USE", "USE", "en", "emb"), ("SBERT-mpnet", "SBERT-mpnet", "en", "emb"),
                                                 ("ALBERT-paraphrase", "ALBERT-paraphrase", "en", "emb"), ("RoBERTa-güncel", "RoBERTa (all-distilroberta-v1)", "en", "emb")]),
        ("Türkçe metin", [("mE5-large-instruct", "mE5-large-instruct", "tr", "emb"), ("BGE-M3", "BGE-M3", "tr", "emb"),
                          ("Qwen3-Emb-4B", "Qwen3-Embedding-4B", "tr", "emb"), ("SBERT çok dilli (TR)", "SBERT çok dilli", "tr", "emb"),
                          ("BERTurk-STS (TR)", "BERTurk-STS", "tr", "emb"), ("TF-IDF (TR)", "TF-IDF (Türkçe)", "tr", "trtf")]),
        ("Statik vektörler ve ince ayarsız kodlayıcılar (ablasyon)", [("Word2Vec", "Word2Vec", "en", "emb"), ("FastText", "FastText", "en", "emb"),
                          ("GloVe", "GloVe", "en", "emb"), ("BERT", "BERT (NLI)", "en", "emb"), ("RoBERTa", "RoBERTa (NLI)", "en", "emb"),
                          ("BERT-raw", "BERT-raw", "en", "emb"), ("RoBERTa-raw", "RoBERTa-raw", "en", "emb"), ("ALBERT", "ALBERT", "en", "emb")]),
        ("Sözcüksel ve basit temel çizgiler", [("Temel: TF-IDF", "TF-IDF", "en", "emb"), ("Temel: Çoğunluk", "Çoğunluk", "en", "base"),
                                               ("Temel: Rastgele", "Rastgele", "en", "base")]),
    ]
    n = sum(len(r) for _, r in groups) + len(groups)
    fig, ax = plt.subplots(figsize=(WL, (n * 0.36 + 1.7) * CM))
    y = 0; yt, yl = [], []
    heads = []
    for g, rows in groups:
        yt.append(y); yl.append(g); heads.append(len(yt) - 1)
        y -= 1
        for m, lab, lang, kind in rows:
            pts = []
            if kind == "llm":
                r = row(m, "sifir_atis/en"); pts.append((r, "o", VERM, VERM))
            elif kind == "base":
                r = row(m, "en/plain/ad"); pts.append((r, "o", "#555555", "#555555"))
            elif kind == "trtf":
                for nm, face in (("ad", ORANGE), ("ad+tanim", "white")):
                    r = TRS[TRS.kosul == f"tr/stem/{nm}"].iloc[0]
                    pts.append((pd.Series({"hit1": r.hit1, "hit1_ga_alt": r.ga_alt, "hit1_ga_ust": r.ga_ust}), "o", ORANGE, face))
            else:
                col = PURPLE if lang == "tr" else (GREEN if m in ("Qwen3-Emb-4B", "mE5-large-instruct", "BGE-M3", "Qwen3-Emb-0.6B") else BLUE)
                if m in ("Word2Vec", "FastText", "GloVe", "BERT", "RoBERTa", "BERT-raw", "RoBERTa-raw", "ALBERT"):
                    col = GREY
                if m == "Temel: TF-IDF":
                    col = ORANGE
                for nm, face in (("ad", col), ("ad+tanim", "white")):
                    r = best(m, lang, nm)
                    if r is not None:
                        pts.append((r, "o", col, face))
            for r, mm, ec, fc in pts:
                if not np.isnan(r.hit1_ga_alt):
                    ax.plot([r.hit1_ga_alt, r.hit1_ga_ust], [y, y], color=ec, lw=0.8, zorder=2)
                ax.plot(r.hit1, y, marker=mm, ms=4.6, mfc=fc, mec=ec, mew=0.9, ls="", zorder=3)
            if len(pts) == 2:
                a, b = pts[0][0].hit1, pts[1][0].hit1
                ax.plot([min(a, b), max(a, b)], [y, y], color="#BBBBBB", lw=0.6, zorder=1)
            yt.append(y); yl.append(lab); y -= 1
        y -= 0.25
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=8); ax.set_ylim(y + 0.6, 0.6)
    for k, tl in enumerate(ax.get_yticklabels()):
        if k in heads:
            tl.set_fontstyle("italic"); tl.set_fontweight("bold"); tl.set_color("#333333")
    ax.set_xlim(0, 1.0); ax.xaxis.set_major_formatter(TRD); ax.set_xlabel("Mesaj düzeyi isabet (hit@1), N = 2.000; çizgiler %95 önyükleme GA")
    ax.grid(axis="x", color="#E6E6E6", lw=0.5, zorder=0); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], marker="o", color="#555555", mfc="#555555", ls="", ms=5, label="Yalnız ihtiyaç adı"),
          Line2D([], [], marker="o", color="#555555", mfc="white", ls="", ms=5, label="İhtiyaç adı + kısa tanım")]
    ax.legend(handles=hs, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_all.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 5: açık ve örtük (dumbbell)
def fig_explicit():
    rows = [("Temel: TF-IDF", "en/stem/ad", "TF-IDF (gövde)", ORANGE), ("Temel: TF-IDF", "en/stem/ad+tanim", "TF-IDF (gövde) + tanım", ORANGE),
            ("USE", "en/plain/ad", "USE", BLUE), ("USE", "en/plain/ad+tanim", "USE + tanım", BLUE),
            ("SBERT-mpnet", "en/plain/ad", "SBERT-mpnet", BLUE), ("SBERT-mpnet", "en/plain/ad+tanim", "SBERT-mpnet + tanım", BLUE),
            ("mE5-large-instruct", "en/plain/ad", "mE5-large-instruct", GREEN), ("mE5-large-instruct", "en/plain/ad+tanim", "mE5-large-instruct + tanım", GREEN),
            ("Qwen3-Emb-4B", "en/plain/ad", "Qwen3-Embedding-4B", GREEN), ("Qwen3-Emb-4B", "en/plain/ad+tanim", "Qwen3-Embedding-4B + tanım", GREEN)] + \
           [(f"BDM {m}", "sifir_atis/en", lab, VERM) for m, lab in reversed(LLM)]
    fig, ax = plt.subplots(figsize=(WL, (len(rows) * 0.42 + 1.5) * CM))
    for i, (m, c, lab, col) in enumerate(rows):
        y = -i - (0.35 if i >= 10 else 0)
        r = row(m, c); e, im = r.hit1_explicit, r.hit1_implicit
        ax.plot([im, e], [y, y], color="#C8C8C8", lw=1.6, zorder=1, solid_capstyle="round")
        ax.plot(e, y, "o", ms=5, color=BLUE, mec="white", mew=0.4, zorder=3)
        ax.plot(im, y, "o", ms=5, color=GREEN, mec="white", mew=0.4, zorder=3)
        ax.text(im - 0.015, y, f2(im), ha="right", va="center", fontsize=7.3, color="#333333")
        ax.text(e + 0.015, y, f2(e), ha="left", va="center", fontsize=7.3, color="#333333")
    ys = [-i - (0.35 if i >= 10 else 0) for i in range(len(rows))]
    ax.set_yticks(ys); ax.set_yticklabels([r[2] for r in rows], fontsize=8)
    ax.axhline(-9.65, color="#DDDDDD", lw=0.6)
    ax.set_xlim(0, 1.06); ax.set_ylim(min(ys) - 0.7, 0.7); ax.xaxis.set_major_formatter(TRD)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_xlabel("hit@1 (açık anlatım N = 1.287; örtük anlatım N = 592)")
    ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0); ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
    hs = [Line2D([], [], marker="o", color=BLUE, ls="", ms=5, label="Açık anlatım"), Line2D([], [], marker="o", color=GREEN, ls="", ms=5, label="Örtük anlatım")]
    ax.legend(handles=hs, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_explicit.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 6: üretici modelin etkisi
def fig_generator():
    T = pd.read_csv(f"{O2}/uretici_yontem_anlatim.csv"); L = pd.read_csv(f"{O2}/uretici_sozcuksel_sizinti.csv")
    gens = [("claude", "Claude Opus 5.5\nmetinleri"), ("gemini", "Gemini 3.8 Flash\nmetinleri"), ("gpt", "GPT-5 mini\nmetinleri")]
    llm_m = [f"BDM {m}" for m, _ in LLM]
    fig, axes = plt.subplots(1, 2, figsize=(WL, 6.4 * CM), sharey=True)
    for ax, (e, title, n) in zip(axes, (("explicit", "(a) Açık anlatım", 432), ("implicit", "(b) Örtük anlatım", 150))):
        series = [("Atanan ihtiyacın adı metinde geçiyor", GREY, [L[(L.uretici == g) & (L.anlatim == e)].ad_gecme_orani.iloc[0] for g, _ in gens]),
                  ("TF-IDF (gövde)", ORANGE, [T[(T.yontem == "TF-IDF (gövde)") & (T.uretici == g)][f"hit1_{e}"].iloc[0] for g, _ in gens]),
                  ("USE (düz)", BLUE, [T[(T.yontem == "USE (düz)") & (T.uretici == g)][f"hit1_{e}"].iloc[0] for g, _ in gens]),
                  ("Beş BDM'nin ortalaması", VERM, [T[(T.yontem.isin(llm_m)) & (T.uretici == g)][f"hit1_{e}"].mean() for g, _ in gens])]
        x = np.arange(len(gens)); w = 0.19
        for k, (lab, col, vals) in enumerate(series):
            bars = ax.bar(x + (k - 1.5) * w, vals, w * 0.92, color=col, label=lab, zorder=2)
            for b_, v in zip(bars, vals):
                ax.text(b_.get_x() + b_.get_width() / 2, v + 0.015, f2(v), ha="center", va="bottom", fontsize=6.4, rotation=90)
        ax.set_xticks(x); ax.set_xticklabels([g[1] for g in gens], fontsize=7.8)
        ax.set_title(f"{title} (her üreticide N = {n})", fontsize=8.5, loc="left")
        ax.set_ylim(0, 1.18); ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0]); ax.yaxis.set_major_formatter(TRD)
        ax.grid(axis="y", color="#EEEEEE", lw=0.5, zorder=0)
    axes[0].set_ylabel("Oran / hit@1")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.01), fontsize=7.8, handlelength=1.0, columnspacing=1.0)
    fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(f"{OUT}/fig_generators.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 7: kademeli tasarım
def fig_cascade():
    C = pd.read_csv(f"{O2}/kademeli_egriler.csv"); CV = pd.read_csv(f"{O2}/kademeli_capraz_dogrulama.csv")
    ORC = pd.read_csv(f"{O2}/kademeli_anlatim_kehaneti.csv")
    fig, ax = plt.subplots(figsize=(WL, 7.2 * CM))
    spec = [("TF-IDF (gövde, ad)", "claude-opus-5-5", VERM, "-", "TF-IDF → Claude Opus 5.5"),
            ("TF-IDF (gövde, ad)", "gemini-3.8-flash", BLUE, "-", "TF-IDF → Gemini 3.8 Flash"),
            ("Qwen3-Emb-4B (düz, ad+tanım)", "claude-opus-5-5", VERM, "--", "Qwen3-Embedding-4B + tanım → Claude Opus 5.5"),
            ("Qwen3-Emb-4B (düz, ad+tanım)", "gemini-3.8-flash", BLUE, "--", "Qwen3-Embedding-4B + tanım → Gemini 3.8 Flash")]
    for fs, m, col, ls, lab in spec:
        c = C[(C.ilk_kademe == fs) & (C.bdm == m) & (C.guven == "s1")].sort_values("bdm_payi")
        ax.plot(c.bdm_payi, c.hit1, color=col, ls=ls, lw=1.3, label=lab, zorder=2)
    for fs, m, col, mk in [("TF-IDF (gövde, ad)", "claude-opus-5-5", VERM, "o"), ("TF-IDF (gövde, ad)", "gemini-3.8-flash", BLUE, "o")]:
        r = CV[(CV.ilk_kademe == fs) & (CV.bdm == m) & (CV.guven == "s1") & (CV.politika == "hedef")].iloc[0]
        ax.plot(r.bdm_payi_ort, r.hit1_ort, marker=mk, ms=6, color=col, mec="black", mew=0.6, zorder=4)
        txt = f"Çapraz doğrulamayla seçilen eşik\n%{r.bdm_payi_ort * 100:.0f} BDM payı, hit@1 {f2(r.hit1_ort, 3)}"
        xy_t = (0.56, 0.842) if m.startswith("claude") else (0.035, 0.952)
        ax.annotate(txt, (r.bdm_payi_ort, r.hit1_ort), xytext=xy_t, fontsize=7.3, color=col,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=col))
    o = ORC[(ORC.ilk_kademe == "TF-IDF (gövde, ad)") & (ORC.bdm == "claude-opus-5-5")].iloc[0]
    ax.plot(o.bdm_payi, o.hit1, marker="*", ms=8, color="white", mec="black", mew=0.7, zorder=5)
    ax.annotate("Karma ve örtük paylaşımların tümü BDM'ye\n(anlatım biçimi bilindiği varsayımı)", (o.bdm_payi, o.hit1), xytext=(0.56, 0.893), fontsize=7.3,
                arrowprops=dict(arrowstyle="-", lw=0.5, color="#444444"))
    llm_op = C[(C.bdm == "claude-opus-5-5") & (C.bdm_payi == 1.0)].hit1.iloc[0]
    ax.axhline(llm_op, color="#777777", ls=":", lw=0.9)
    ax.text(1.0, llm_op + 0.004, f"Claude Opus 5.5 tek başına {f2(llm_op, 3)}", ha="right", va="bottom", fontsize=7.3, color="#444444")
    tf0 = C[(C.ilk_kademe == "TF-IDF (gövde, ad)") & (C.bdm_payi == 0)].hit1.iloc[0]
    ax.set_xlim(0, 1.0); ax.set_ylim(0.70, 0.975); ax.xaxis.set_major_formatter(PCT); ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f2(v)))
    ax.set_xlabel("BDM'ye yönlendirilen paylaşımların payı (en düşük benzerlik puanlılardan başlayarak)")
    ax.set_ylabel("Kademeli sistemin hit@1 değeri")
    ax.grid(color="#EEEEEE", lw=0.5, zorder=0)
    ax.legend(frameon=False, loc="lower right", fontsize=7.6, bbox_to_anchor=(1.0, 0.02))
    fig.tight_layout(); fig.savefig(f"{OUT}/fig_cascade.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 8: gerçek veri
def fig_real():
    R = pd.read_csv(f"{O2}/gercek_kapali_kume.csv")
    def get(v, m, c):
        r = R[(R.veri == v) & (R.yontem == m) & (R.kosul == c)]
        assert len(r) == 1, (v, m, c); return r.iloc[0]
    en = [(f"BDM {m}", "sifir_atis", lab, VERM) for m, lab in LLM] + [
          ("Temel: TF-IDF", "en/stem/ad", "TF-IDF (gövde)", ORANGE), ("Qwen3-Emb-4B", "en/plain/ad+tanim", "Qwen3-Embedding-4B + tanım", GREEN),
          ("mE5-large-instruct", "en/plain/ad+tanim", "mE5-large-instruct + tanım", GREEN), ("BGE-M3", "en/plain/ad", "BGE-M3", GREEN),
          ("USE", "en/plain/ad", "USE", BLUE), ("SBERT-mpnet", "en/plain/ad", "SBERT-mpnet", BLUE),
          ("SetFit-EN", "tam_sentetik", "SetFit (2.000 sentetik paylaşımla)", SKY), ("Temel: Çoğunluk", "en/plain/ad", "Çoğunluk (kehanet)", "#555555")]
    tr = [(f"BDM {m}", "sifir_atis", lab, VERM) for m, lab in LLM] + [
          ("TF-IDF (TR)", "tr/stem/ad", "TF-IDF (Türkçe, gövde)", ORANGE), ("mE5-large-instruct", "tr/raw/ad+tanim", "mE5-large-instruct + tanım", PURPLE),
          ("BGE-M3", "tr/raw/ad+tanim", "BGE-M3 + tanım", PURPLE), ("Qwen3-Emb-4B", "tr/raw/ad+tanim", "Qwen3-Embedding-4B + tanım", PURPLE),
          ("SBERT çok dilli (TR)", "tr/plain/ad", "SBERT çok dilli", PURPLE), ("BERTurk-STS (TR)", "tr/plain/ad", "BERTurk-STS", PURPLE),
          ("SetFit-TR", "tam_sentetik", "SetFit (2.000 sentetik paylaşımla)", SKY), ("Temel: Çoğunluk", "en/plain/ad", "Çoğunluk (kehanet)", "#555555")]
    fig, axes = plt.subplots(1, 2, figsize=(WL, 7.6 * CM))
    for ax, (v, rows, title) in zip(axes, (("gercek_en", en, "(a) İngilizce, HumAID (N = 30)"), ("gercek_tr", tr, "(b) Türkçe, Kaggle (N = 21)"))):
        for i, (m, c, lab, col) in enumerate(rows):
            r = get(v, m, c); y = -i
            ax.plot([r.ga_alt, r.ga_ust], [y, y], color=col, lw=0.9, zorder=2)
            ax.plot(r.hit1, y, "o", ms=4.6, color=col, mec="white", mew=0.4, zorder=3)
        ax.set_yticks([-i for i in range(len(rows))]); ax.set_yticklabels([r[2] for r in rows], fontsize=7.6)
        ax.set_xlim(0, 1.04); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f2(v))); ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax.set_title(title, fontsize=8.5, loc="left"); ax.grid(axis="x", color="#EEEEEE", lw=0.5, zorder=0)
        ax.spines["left"].set_visible(False); ax.tick_params(axis="y", length=0)
        ax.set_xlabel("hit@1 (kesin binom %95 GA)", fontsize=8)
    fig.tight_layout(w_pad=1.2); fig.savefig(f"{OUT}/fig_real_posts.png", dpi=600); plt.close(fig)

# ------------------------------------------------------------------ Şekil 1: genel çerçeve
def _box(ax, x, y, w, h, fc, title, body, fs=8.2, fs2=7.4):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2", fc=fc, ec="#1F4E79", lw=0.8))
    lines = [(title, fs, True)] + [(b, fs2, False) for b in body]
    PT = 100 / (FIG_H_CM / 2.54 * 72)
    hs = [f * 1.25 * PT for _, f, _ in lines]; tot = sum(hs) + 0.8
    yy = y + h / 2 + tot / 2
    for (t, f, b), hh in zip(lines, hs):
        ax.text(x + w / 2, yy - hh / 2, t, ha="center", va="center", fontsize=f, fontweight="bold" if b else "normal")
        yy -= hh + (0.8 if b else 0)

def _arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=8, lw=0.9, color="#1F4E79", shrinkA=0, shrinkB=0))

FIG_H_CM = 11.6
def fig_framework():
    fig = plt.figure(figsize=(WL, FIG_H_CM * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")
    C1, C2, C3 = "#D6E2F0", "#FBE5D6", "#E2EFDA"
    def band(y, txt):
        ax.text(0.3, y, txt, rotation=90, ha="left", va="center", fontsize=8.2, fontweight="bold", color="#1F4E79")
    xs = [(4.0, 30.2), (36.4, 30.2), (68.8, 30.2)]
    # veri
    band(86, "Veri")
    _box(ax, *xs[0][:1], 74, xs[0][1], 23.5, C1, "Simüle veri seti", ["etiket öncelikli üretim: Claude Opus 5.5", "denetim: Claude Sonnet 5.5",
         "2.000 paylaşım, 39 ilçe, Türkçe/İngilizce", "açık / karma / örtük anlatım, 30 persona"])
    _box(ax, xs[1][0], 74, xs[1][1], 23.5, C1, "Alternatif üreticiler", ["aynı 617 plan yeniden yazıldı:", "Gemini 3.8 Flash ve GPT-5 mini",
         "farklı aileden modelle denetim", "kendi ailesini kayırma sınaması"])
    _box(ax, xs[2][0], 74, xs[2][1], 23.5, C1, "Gerçek paylaşımlar", ["HumAID (İngilizce, 207 paylaşım)", "Kaggle (Türkçe, 403 paylaşım)",
         "66 madde ve 13 sektörle etiketleme", "HumAID kitle etiketleri (675 tweet)"])
    # ihtiyaç temsili
    ax.add_patch(FancyBboxPatch((4.0, 61.5), 95.0, 8.5, boxstyle="round,pad=0,rounding_size=1.2", fc="#F2F2F2", ec="#1F4E79", lw=0.8))
    ax.text(51.5, 67.6, "Ortak ihtiyaç uzayı", ha="center", va="center", fontsize=8.2, fontweight="bold")
    ax.text(51.5, 64.0, "66 madde → 32 yakın anlam grubu → 13 sektör (IASC kümelerine dayalı); gömmelerde ihtiyaç adı ya da ad + kısa tanım", ha="center", va="center", fontsize=7.4)
    for x, w in xs:
        _arrow(ax, x + w / 2, 74, x + w / 2, 70.2)
    # yöntemler
    band(46, "Yöntemler")
    _box(ax, xs[0][0], 33, xs[0][1], 25, C2, "Yerel yöntemler", ["TF-IDF ve cümle kodlayıcılar", "mE5, BGE-M3, Qwen3-Embedding", "kosinüs benzerliği → en yüksek ihtiyaç",
         "ince ayar ablasyonu; SetFit"])
    _box(ax, xs[1][0], 33, xs[1][1], 25, C2, "Büyük dil modelleri", ["Claude Opus, Sonnet, Haiku 5.5;", "GPT-5 mini; Gemini 3.8 Flash",
         "sıfır atış (İngilizce, Türkçe), 66 örnekli,", "çok etiketli, açık küme; üç tekrar"])
    _box(ax, xs[2][0], 33, xs[2][1], 25, C2, "Konum çıkarımı", ["alt dize eşleme", "bulanık ilçe sözlüğü", "Türkçe varlık adı tanıma (NER)", "bozulmuş yazımlarla sınama"])
    for x, w in xs:
        _arrow(ax, x + w / 2, 61.5, x + w / 2, 58.2)
    # değerlendirme
    band(14, "Değerlendirme")
    _box(ax, xs[0][0], 2, xs[0][1], 25, C3, "Ölçütler ve testler", ["hit@1; madde, grup, sektör düzeyi", "mikro/makro F1; ilçe doğruluğu",
         "önyükleme GA; McNemar + Holm", "öz-tercih regresyonu; duyarlılık"])
    _box(ax, xs[1][0], 2, xs[1][1], 25, C3, "Kademeli tasarım", ["TF-IDF benzerliği düşük paylaşımlar", "BDM'ye yönlendirilir", "eşik: 5 katlı çapraz doğrulama",
         "isabet, maliyet ve gecikme"])
    _box(ax, xs[2][0], 2, xs[2][1], 25, C3, "İlçe düzeyinde ihtiyaç haritası", ["her ilçenin en sık ihtiyacı", "ilçe başına paylaşım duyarlılığı",
         "sektör düzeyinde özet", "karar desteği çıktısı"])
    for x, w in xs:
        _arrow(ax, x + w / 2, 33, x + w / 2, 27.2)
    fig.savefig(f"{OUT}/fig_framework.png", dpi=600); plt.close(fig)

if __name__ == "__main__":
    fig_framework()
    fig_text_type(); fig_overall(); fig_explicit(); fig_generator(); fig_cascade(); fig_real()
    print(sorted(os.listdir(OUT)))

# ------------------------------------------------------------------ Grafiksel özet (Türkçe ve İngilizce)
GA_TXT = {
    "tr": dict(b1="Simüle paylaşımlar", b1s="2.000 paylaşım · 39 ilçe\n3 üretici model · TR/EN", b2="Yerel yöntemler",
               b2s="TF-IDF, kodlayıcılar\n(ad + tanım)", b3="Büyük dil modelleri", b3s="5 model, sıfır atış", b4="Kademeli tasarım",
               b4s="TF-IDF güveni düşük\npaylaşımlar BDM'ye", b5="Gerçek paylaşımlar", b5s="HumAID (EN), Kaggle (TR)",
               chart="Anlatım biçimine göre hit@1 (simüle veri)", expl="Açık", impl="Örtük",
               note="Kademeli (TF-IDF → Gemini): BDM payı %{share}, hit@1 {cas} (Gemini tek başına {gem})",
               note2="Gerçek paylaşımlar: BDM'ler {lo}–{hi}; en iyi yerel yöntem {loc}",
               groups=("TF-IDF", "Qwen3-Emb-4B\n+ tanım", "USE", "Claude\nOpus 5.5", "Gemini\n3.8 Flash")),
    "en": dict(b1="Simulated posts", b1s="2,000 posts · 39 districts\n3 generator models · TR/EN", b2="Local methods",
               b2s="TF-IDF, encoders\n(name + description)", b3="Large language models", b3s="5 models, zero-shot", b4="Cascade",
               b4s="low TF-IDF confidence\nposts go to an LLM", b5="Real posts", b5s="HumAID (EN), Kaggle (TR)",
               chart="Hit@1 by phrasing (simulated data)", expl="Explicit", impl="Implicit",
               note="Cascade (TF-IDF → Gemini): {share}% sent to LLM, hit@1 {cas} (Gemini alone {gem})",
               note2="Real posts: LLMs {lo}–{hi}; best local method {loc}",
               groups=("TF-IDF", "Qwen3-Emb-4B\n+ description", "USE", "Claude\nOpus 5.5", "Gemini\n3.8 Flash")),
}
def graphical_abstract(lang):
    T = GA_TXT[lang]
    dec = (lambda v, d=2: f2(v, d)) if lang == "tr" else (lambda v, d=2: f"{v:.{d}f}")
    W, H = 10.0, 7.8
    fig = plt.figure(figsize=(W * CM, H * CM)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 78); ax.axis("off")
    PT = 78 / (H / 2.54 * 72)
    def box(x, y, w, h, fc, t, s, fs=6.4, fs2=5.7):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.3", fc=fc, ec="#1F4E79", lw=0.7))
        lines = [(t, fs, True)] + [(ln, fs2, False) for ln in s.split("\n")]
        hs = [f * 1.22 * PT for _, f, _ in lines]; tot = sum(hs) + 0.7; yy = y + h / 2 + tot / 2
        for (tt, f, b), hh in zip(lines, hs):
            ax.text(x + w / 2, yy - hh / 2, tt, ha="center", va="center", fontsize=f, fontweight="bold" if b else "normal"); yy -= hh + (0.7 if b else 0)
    def arr(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=6, lw=0.8, color="#1F4E79", shrinkA=0, shrinkB=0))
    box(0.6, 57.5, 27.5, 19.5, "#D6E2F0", T["b1"], T["b1s"])
    box(0.6, 45.0, 27.5, 10.5, "#D6E2F0", T["b5"], T["b5s"], fs=6.2, fs2=5.6)
    box(35.0, 63.5, 29.0, 13.5, "#E8EEF7", T["b2"], T["b2s"], fs=6.2)
    box(35.0, 45.0, 29.0, 15.5, "#FBE5D6", T["b3"], T["b3s"], fs=6.2)
    box(71.0, 45.0, 28.4, 32.0, "#E2EFDA", T["b4"], T["b4s"], fs=6.4)
    arr(28.1, 67.0, 35.0, 70.0); arr(28.1, 62.0, 35.0, 54.0); arr(28.1, 50.0, 35.0, 50.0)
    arr(64.0, 70.0, 71.0, 66.0); arr(64.0, 53.0, 71.0, 56.0)
    CV = pd.read_csv(f"{O2}/kademeli_capraz_dogrulama.csv")
    r = CV[(CV.ilk_kademe == "TF-IDF (gövde, ad)") & (CV.guven == "s1") & (CV.bdm == "gemini-3.8-flash") & (CV.politika == "en_iyi")].iloc[0]
    R = pd.read_csv(f"{O2}/gercek_kapali_kume.csv")
    llm_real = R[R.yontem.str.startswith("BDM")].hit1
    loc_en = R[(R.veri == "gercek_en") & (R.yontem == "Temel: TF-IDF") & (R.kosul == "en/stem/ad")].hit1.iloc[0]
    loc_tr = R[(R.veri == "gercek_tr") & (R.yontem == "mE5-large-instruct") & (R.kosul == "tr/raw/ad+tanim")].hit1.iloc[0]
    keys = [("Temel: TF-IDF", "en/stem/ad"), ("Qwen3-Emb-4B", "en/plain/ad+tanim"), ("USE", "en/plain/ad"),
            ("BDM claude-opus-5-5", "sifir_atis/en"), ("BDM gemini-3.8-flash", "sifir_atis/en")]
    ex = [row(*k).hit1_explicit for k in keys]; im = [row(*k).hit1_implicit for k in keys]
    ax.text(0.8, 41.0, T["chart"], ha="left", va="center", fontsize=6.3, fontweight="bold")
    share = f"{r.bdm_payi_ort * 100:.0f}"
    ax.text(0.8, 37.6, T["note"].format(share=share, cas=dec(r.hit1_ort, 3), gem=dec(r.bdm_tek_hit1, 3)), ha="left", va="center", fontsize=5.6)
    ax.text(0.8, 34.6, T["note2"].format(lo=dec(llm_real.min()), hi=dec(llm_real.max()), loc=f"{dec(min(loc_en, loc_tr))}–{dec(max(loc_en, loc_tr))}"), ha="left", va="center", fontsize=5.6)
    axb = fig.add_axes([0.075, 0.075, 0.915, 0.33])
    x = np.arange(len(keys)); w = 0.36
    b1 = axb.bar(x - w / 2, ex, w, color=BLUE, label=T["expl"]); b2 = axb.bar(x + w / 2, im, w, color=GREEN, label=T["impl"])
    for bars in (b1, b2):
        for b_ in bars:
            axb.text(b_.get_x() + b_.get_width() / 2, b_.get_height() + 0.025, dec(b_.get_height()), ha="center", va="bottom", fontsize=5.2)
    axb.set_xticks(x); axb.set_xticklabels(T["groups"], fontsize=5.6)
    axb.set_ylim(0, 1.18); axb.set_yticks([0, 0.5, 1.0]); axb.set_yticklabels([dec(v, 1) for v in (0, 0.5, 1.0)], fontsize=5.6)
    axb.tick_params(length=2, width=0.5, pad=1.5)
    for s_ in ("left", "bottom"):
        axb.spines[s_].set_linewidth(0.5)
    axb.legend(frameon=False, fontsize=5.7, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.13), handlelength=1.0, columnspacing=0.9, handletextpad=0.4)
    fig.savefig(f"{OUT}/grafiksel_ozet_{lang}.png", dpi=600); plt.close(fig)

for _lg in ("tr", "en"):
    graphical_abstract(_lg)
