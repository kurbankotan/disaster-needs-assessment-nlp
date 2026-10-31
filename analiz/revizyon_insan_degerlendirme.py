# -*- coding: utf-8 -*-
"""Simüle paylaşımların etiketlerinin insan değerlendirmesi: örtük paylaşımlarda üretici modelin atadığı ihtiyaçların
iki bağımsız etiketleyicinin kör etiketlemesiyle karşılaştırılması.

Örneklem: üç üreticinin ortak 617 planındaki örtük paylaşımların tamamı (150 paylaşım, Claude Opus 5.5 metni, Türkçe).
Bu paylaşımlar ana veri setinde de yer aldığından tüm yöntemlerin kayıtlı tahminleri ve GPT-5 mini çapraz denetimi vardır.

    python analiz/revizyon_insan_degerlendirme.py olustur <cikti_klasoru>
        İki özdeş boş çalışma kitabı (Etiketleyici 1 ve 2) üretir. Etiketleyiciler üretici etiketlerini görmez.
    python analiz/revizyon_insan_degerlendirme.py puanla <etiketleyici1.xlsx> <etiketleyici2.xlsx>
        Etiketleyiciler arası uyum, insan etiketleri ile üretici etiketlerinin uyumu, yöntemlerin insan etiketlerine göre
        yeniden puanlanması ve BDM hatalarının ayrıştırılması; çıktılar runs/revizyon/results/analiz/ klasörüne yazılır.
"""
import os, sys, random, pickle
from revizyon_ortak import *
from openpyxl import Workbook, load_workbook
from openpyxl.styles import Font, Alignment, PatternFill, Border, Side
from openpyxl.worksheet.datavalidation import DataValidation
from openpyxl.utils import get_column_letter as L

SEED = 42
PREFIX = "H"
N_NEED = 3
LY = "Listede yok"

# ------------------------------------------------------------------------------------------------ örneklem
def sample():
    alt = pd.read_csv(os.path.join(RR, "alt_claude_veri.csv"), encoding="utf-8-sig")
    imp = alt[alt.explicitness == "implicit"].copy()
    assert len(imp) == 150
    main = D.set_index("id")
    rows = []
    for r in imp.itertuples():
        assert main.loc[r.id, "Message"] == r.Message           # ana veri setindeki metnin aynısı
        rows.append({"id": int(r.id), "metin": r.Message, "denetim_gpt5mini": bool(r.qc_ok)})
    rows.sort(key=lambda x: x["id"])
    random.Random(SEED).shuffle(rows)
    for i, x in enumerate(rows, 1):
        x["sira"] = f"{PREFIX}-{i:03d}"
    return rows

# ------------------------------------------------------------------------------------------------ çalışma kitabı
FONT = "Arial"
F_BODY = Font(name=FONT, size=10); F_BOLD = Font(name=FONT, size=10, bold=True)
F_HEAD = Font(name=FONT, size=10, bold=True, color="FFFFFF"); F_TITLE = Font(name=FONT, size=13, bold=True)
F_NOTE = Font(name=FONT, size=9, italic=True, color="595959")
FILL_HEAD = PatternFill("solid", fgColor="1F4E79"); FILL_INPUT = PatternFill("solid", fgColor="FFF2CC")
FILL_EX = PatternFill("solid", fgColor="F2F2F2")
THIN = Side(style="thin", color="BFBFBF"); BOX = Border(left=THIN, right=THIN, top=THIN, bottom=THIN)
WRAP = Alignment(wrap_text=True, vertical="top")
HEADS = ["Sıra", "Paylaşım metni", "Somut ihtiyaç var mı?", "İhtiyaç 1 (en önemli)", "İhtiyaç 2", "İhtiyaç 3", "Anlatım biçimi",
         "Listede olmayan ihtiyaç", "Emin değilim", "Not"]
WIDTHS = [8, 72, 13, 26, 26, 26, 12, 22, 10, 28]

def need_value(tr):
    return f"{tr} | {TR2EN[tr]}"

def instructions(ws, n):
    ws.title = "Talimatlar"
    ws.column_dimensions["A"].width = 30
    for c in "BCDEFG":
        ws.column_dimensions[c].width = 24
    ws.page_setup.orientation = "landscape"; ws.sheet_properties.pageSetUpPr.fitToPage = True
    ws.page_setup.fitToWidth = 1; ws.page_setup.fitToHeight = 0
    r = 1
    ws.cell(r, 1, "Simüle deprem paylaşımlarında ihtiyaçların değerlendirilmesi").font = F_TITLE; r += 2
    lines = [
        ("Görev", "Her paylaşımı okuyun ve paylaşımı yazan kişinin ya da sözünü ettiği kişilerin ihtiyaç duyduğu şeyleri 66 maddelik listeden seçin. "
                  "Paylaşımlar bir dil modeliyle üretilmiş simüle metinlerdir ve çoğu ihtiyacı adıyla söylemez, durumu anlatır. İhtiyacı metindeki "
                  "ipuçlarından çıkarın; metinde dayanağı olmayan, yalnızca genel olarak olası ihtiyaçları eklemeyin."),
        ("Doldurulacak hücreler", "'Paylaşımlar' sayfasında yalnızca sarı sütunları doldurun: 'Somut ihtiyaç var mı?', 'İhtiyaç 1-3', 'Anlatım biçimi', "
                                  "'Listede olmayan ihtiyaç', 'Emin değilim' ve 'Not'. Öteki sütunlara dokunmayın. Açılır listeli hücrelerde listeden seçin."),
        ("Somut ihtiyaç var mı?", "Paylaşımdan bir malzeme, hizmet ya da ekip ihtiyacı makul bir güvenle çıkarılabiliyorsa 'Evet', çıkarılamıyorsa 'Hayır' seçin. "
                                  "'Hayır' satırlarında öteki sarı sütunları boş bırakın."),
        ("İhtiyaç seçimi", "En fazla üç ihtiyaç seçin. 'İhtiyaç 1' sütununa paylaşıma göre en önemli ya da en acil ihtiyacı yazın, ötekileri önem sırasıyla ekleyin; "
                           "anlatılan durum birden çok ihtiyaca açıkça işaret ediyorsa hepsini seçin. "
                           "Aynı ihtiyacı iki kez seçmeyin. Listede karşılığı olmayan bir ihtiyaç için 'Listede yok' seçin ve adını 'Listede olmayan ihtiyaç' "
                           "sütununa yazın. Bir ihtiyacı listedeki yakın bir maddeye zorlamayın."),
        ("Yakın anlamlı maddeler", "Gıda / kuru gıda / konserve / sıcak yemek / ekmek; fener / el feneri / aydınlatıcı; ilaç ve ilaç türleri gibi gruplarda "
                                   "metindeki ifadeye en uygun maddeyi seçin. Metin yalnızca genel bir durum anlatıyorsa (ör. 'iki gündür bir şey yemedik') genel "
                                   "maddeyi seçin. 'İhtiyaç listesi' sayfasındaki 'Yakın anlam grubu' sütunu bu grupları gösterir."),
        ("Anlatım biçimi", "Açık: ihtiyaç adıyla ya da gündelik eş anlamlısıyla anılıyor. Örtük: ihtiyaç adlandırılmıyor, anlatılan durumdan çıkarılıyor "
                           "(ör. iğneleri enkazda kalan diyabet hastası için insülin). Karma: ihtiyaçlardan en az biri açık, en az biri örtük."),
        ("Belirsizlik", "Karar veremediğiniz satırlarda en olası seçimi yapın, 'Emin değilim' sütununda 'Evet' seçin ve kısa bir not düşün."),
        ("Bağımsızlık", "İki etiketleyici birbirine danışmadan, ayrı dosyalarda çalışır. Yayımlanmış veri setlerine, dil modellerine ya da başka kaynaklara "
                        "bakılmaz. Dosya bittikten sonra değiştirilmez; iki etiketleme ayrı ayrı analiz edilir."),
        ("Süre", f"{n} paylaşım vardır; paylaşım başına yaklaşık yarım dakika ile bir dakika sürer. Ara verirken dosyayı kaydetmeniz yeterlidir."),
    ]
    for k, v in lines:
        ws.cell(r, 1, k).font = F_BOLD; ws.cell(r, 1).alignment = WRAP
        c = ws.cell(r, 2, v); c.font = F_BODY; c.alignment = WRAP
        ws.merge_cells(start_row=r, start_column=2, end_row=r, end_column=7)
        ws.row_dimensions[r].height = 15 * max(2, len(v) // 115 + 1)
        r += 1
    r += 1
    ws.cell(r, 1, "Örnek satırlar (uydurma, yalnızca biçimi göstermek için)").font = F_BOLD; r += 1
    hdr = ["Paylaşım", "Somut ihtiyaç var mı?", "İhtiyaç 1 (en önemli)", "İhtiyaç 2", "Anlatım biçimi", "Listede olmayan ihtiyaç"]
    for j, h in enumerate(hdr, 1):
        c = ws.cell(r, j, h); c.font = F_HEAD; c.fill = FILL_HEAD; c.alignment = WRAP; c.border = BOX
    r += 1
    ex = [["Okulun bahçesinde 30 kişi kalıyoruz, gece için çadır ve battaniye lazım.", "Evet", need_value("çadır"), need_value("battaniye"), "Açık", ""],
          ["Annem şeker hastası, iğneleri enkazda kaldı, ne yapacağımızı bilmiyoruz.", "Evet", need_value("insülin"), "", "Örtük", ""],
          ["Köydeki jeneratörün yakıtı bitti, acil mazot gerekiyor.", "Evet", LY, "", "Açık", "mazot"],
          ["Herkese geçmiş olsun, bölgedeki tüm ekiplere kolaylıklar diliyorum.", "Hayır", "", "", "", ""]]
    for row in ex:
        for j, v in enumerate(row, 1):
            c = ws.cell(r, j, v); c.font = F_BODY; c.fill = FILL_EX; c.alignment = WRAP; c.border = BOX
        ws.row_dimensions[r].height = 45
        r += 1
    r += 1
    ws.cell(r, 1, "İlerleme").font = F_BOLD
    ws.cell(r, 2, f"=COUNTA('Paylaşımlar'!C2:C{n + 1})").font = F_BODY
    ws.cell(r, 3, f"/ {n} satır").font = F_BODY; r += 1
    ws.cell(r, 1, "İlerleme sayısı 'Somut ihtiyaç var mı?' sütunundaki dolu hücreleri sayar.").font = F_NOTE

def data_sheet(wb, rows, n_list):
    ws = wb.create_sheet("Paylaşımlar")
    for j, (h, w) in enumerate(zip(HEADS, WIDTHS), 1):
        c = ws.cell(1, j, h); c.font = F_HEAD; c.fill = FILL_HEAD; c.alignment = Alignment(wrap_text=True, vertical="center"); c.border = BOX
        ws.column_dimensions[L(j)].width = w
    ws.row_dimensions[1].height = 30
    for i, x in enumerate(rows, 2):
        for j, v in enumerate([x["sira"], x["metin"]], 1):
            c = ws.cell(i, j, v); c.font = F_BODY; c.alignment = WRAP; c.border = BOX
        for j in range(3, len(HEADS) + 1):
            c = ws.cell(i, j); c.fill = FILL_INPUT; c.border = BOX; c.font = F_BODY; c.alignment = WRAP
        ws.row_dimensions[i].height = 15 * max(2, min(9, len(x["metin"]) // 75 + 1))
    n = len(rows)
    dv_yn = DataValidation(type="list", formula1='"Evet,Hayır"', allow_blank=True)
    dv_need = DataValidation(type="list", formula1=f"='İhtiyaç listesi'!$E$2:$E${n_list + 2}", allow_blank=True)
    dv_exp = DataValidation(type="list", formula1='"Açık,Karma,Örtük"', allow_blank=True)
    dv_uns = DataValidation(type="list", formula1='"Evet"', allow_blank=True)
    for dv in (dv_yn, dv_need, dv_exp, dv_uns):
        dv.error = "Lütfen listeden seçin."; dv.errorTitle = "Geçersiz değer"; dv.showErrorMessage = True
        ws.add_data_validation(dv)
    dv_yn.add(f"C2:C{n + 1}"); dv_need.add(f"D2:F{n + 1}"); dv_exp.add(f"G2:G{n + 1}"); dv_uns.add(f"I2:I{n + 1}")
    ws.freeze_panes = "C2"
    ws.auto_filter.ref = f"A1:{L(len(HEADS))}{n + 1}"

def need_sheet(wb):
    ws = wb.create_sheet("İhtiyaç listesi")
    heads = ["No", "Türkçe", "İngilizce", "Yakın anlam grubu", "Açılır listedeki değer"]
    for j, (h, w) in enumerate(zip(heads, [6, 26, 30, 28, 50]), 1):
        c = ws.cell(1, j, h); c.font = F_HEAD; c.fill = FILL_HEAD; c.border = BOX
        ws.column_dimensions[L(j)].width = w
    trs = list(NL.tr)
    for i, tr in enumerate(trs, 2):
        grp = GROUP_OF_TR[tr] if GROUP_OF_TR[tr] != tr else ""
        for j, v in enumerate([i - 1, tr, TR2EN[tr], grp, need_value(tr)], 1):
            c = ws.cell(i, j, v); c.font = F_BODY; c.border = BOX
    i = len(trs) + 2
    for j, v in enumerate(["", LY, "Not in the list", "", LY], 1):
        c = ws.cell(i, j, v); c.font = F_BODY; c.border = BOX
    ws.freeze_panes = "A2"
    return len(trs)

def olustur(outdir):
    os.makedirs(outdir, exist_ok=True)
    rows = sample()
    paths = []
    for k in (1, 2):
        wb = Workbook()
        instructions(wb.active, len(rows))
        n_list = need_sheet(wb)
        data_sheet(wb, rows, n_list)
        wb.move_sheet("Paylaşımlar", offset=-1)                  # sıra: Talimatlar, Paylaşımlar, İhtiyaç listesi
        wb.properties.creator = ""; wb.properties.lastModifiedBy = ""
        p = os.path.join(outdir, f"Insan_Degerlendirme_Etiketleyici_{k}.xlsx")
        wb.save(p); paths.append(p)
    pd.DataFrame([{"sira": x["sira"], "id": x["id"]} for x in rows]).to_csv(
        os.path.join(outdir, "insan_degerlendirme_sira_kimlik.csv"), index=False, encoding="utf-8-sig")
    print(len(rows), "paylaşım;", *paths, sep="\n  ")
    return paths

# ------------------------------------------------------------------------------------------------ okuma
def load(p):
    wb = load_workbook(p, data_only=True)
    ws = wb["Paylaşımlar"]
    hdr = [c.value for c in ws[1]]
    assert hdr[:len(HEADS)] == HEADS, hdr
    out = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        v = lambda j: (str(row[j]).strip() if row[j] is not None else "")
        needs = [v(j) for j in (3, 4, 5) if v(j)]
        tr = [x.split(" | ")[0].strip() for x in needs if x != LY]
        bad = [x for x in tr if x not in TR2EN]
        assert not bad, (row[0], bad)
        out.append({"sira": v(0), "metin": v(1), "evet": v(2), "ihtiyac_tr": list(dict.fromkeys(tr)), "listede_yok": LY in needs,
                    "bicim": v(6), "listede_olmayan": v(7), "emin_degil": v(8) == "Evet", "not": v(9)})
    return pd.DataFrame(out)

# ------------------------------------------------------------------------------------------------ uyum ölçüleri
def cohen_kappa(x, y):
    x, y = list(x), list(y); n = len(x); cats = sorted(set(x) | set(y))
    po = sum(a == b for a, b in zip(x, y)) / n
    pe = sum((x.count(c) / n) * (y.count(c) / n) for c in cats)
    return po, ((po - pe) / (1 - pe) if pe < 1 else float("nan"))

def masi(a, b):
    if not a and not b:
        return 0.0
    j = len(a & b) / len(a | b)
    m = 1.0 if a == b else (0.67 if (a <= b or b <= a) else (0.33 if a & b else 0.0))
    return 1.0 - j * m

def kripp_alpha_sets(A, B):
    """İki kodlayıcı, küme değerli veri, MASI uzaklığıyla Krippendorff alfa (gerçek paylaşımlardaki hesapla aynı)."""
    vals = list(A) + list(B); n_tot = len(vals)
    do = 2.0 * sum(masi(a, b) for a, b in zip(A, B))
    de = sum(masi(vals[i], vals[j]) for i in range(n_tot) for j in range(n_tot) if i != j)
    return 1.0 - (n_tot - 1) * do / de if de > 0 else float("nan")

def boot_alpha(A, B, reps=1000, seed=42):
    rng = np.random.default_rng(seed); n = len(A); v = []
    for _ in range(reps):
        ix = rng.integers(0, n, n)
        a = kripp_alpha_sets([A[i] for i in ix], [B[i] for i in ix])
        if a == a:
            v.append(a)
    return float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))

def level(S_, lv):
    if lv == "madde":
        return frozenset(S_)
    f = GROUP_OF_TR if lv == "grup" else ITEM_SECTOR
    return frozenset(f[x] for x in S_)

def pair_stats(A, B, lv):
    a = [level(x, lv) for x in A]; b = [level(x, lv) for x in B]
    both = [i for i in range(len(a)) if a[i] and b[i]]
    lo, hi = boot_alpha(a, b)
    return {"alfa_masi": kripp_alpha_sets(a, b), "alfa_ga": [lo, hi], "ayni_kume": float(np.mean([a[i] == b[i] for i in range(len(a))])),
            "en_az_bir_ortak": float(np.mean([bool(a[i] & b[i]) for i in both])) if both else float("nan"),
            "jaccard": float(np.mean([len(a[i] & b[i]) / len(a[i] | b[i]) for i in both])) if both else float("nan")}

# ------------------------------------------------------------------------------------------------ puanlama
def puanla(p1, p2):
    rows = sample()
    a, b = load(p1), load(p2)
    for x in (a, b):
        assert list(x.sira) == [r["sira"] for r in rows], "sıra kodları örneklemle uyuşmuyor"
        assert list(x.metin) == [r["metin"] for r in rows], "metinler örneklemle uyuşmuyor"
        assert set(x.evet) <= {"Evet", "Hayır"}, f"boş ya da geçersiz karar: {sorted(set(x.evet) - {'Evet', 'Hayır'})}"
    ids = [r["id"] for r in rows]
    aud = np.array([r["denetim_gpt5mini"] for r in rows])
    Dm = D.set_index("id")
    plan = [frozenset(x for x in Dm.loc[i, ["label1", "label2", "label3"]] if isinstance(x, str) and x) for i in ids]
    E1 = [frozenset(s) if e == "Evet" else frozenset() for s, e in zip(a.ihtiyac_tr, a.evet)]
    E2 = [frozenset(s) if e == "Evet" else frozenset() for s, e in zip(b.ihtiyac_tr, b.evet)]
    R = {"n": len(rows), "evet_E1": int((a.evet == "Evet").sum()), "evet_E2": int((b.evet == "Evet").sum())}
    po, k = cohen_kappa(a.evet, b.evet); R["karar_uyum"] = po; R["karar_kappa"] = k
    R["emin_degil_E1"] = int(a.emin_degil.sum()); R["emin_degil_E2"] = int(b.emin_degil.sum())
    R["listede_yok_E1"] = int(a.listede_yok.sum()); R["listede_yok_E2"] = int(b.listede_yok.sum())
    R["ortuk_payi_E1"] = float((a.bicim == "Örtük").mean()); R["ortuk_payi_E2"] = float((b.bicim == "Örtük").mean())
    R["karma_payi_E1"] = float((a.bicim == "Karma").mean()); R["karma_payi_E2"] = float((b.bicim == "Karma").mean())
    R["acik_payi_E1"] = float((a.bicim == "Açık").mean()); R["acik_payi_E2"] = float((b.bicim == "Açık").mean())
    for lv in ("madde", "grup", "sektor"):
        R[f"E1_E2_{lv}"] = pair_stats(E1, E2, lv)
        R[f"plan_E1_{lv}"] = pair_stats(plan, E1, lv)
        R[f"plan_E2_{lv}"] = pair_stats(plan, E2, lv)
    # uzlaşı: iki etiketleyicinin madde kümesi aynı (ve boş değil)
    cons = np.array([bool(x) and x == y for x, y in zip(E1, E2)])
    R["uzlasi_n"] = int(cons.sum())
    R["uzlasi_plan_ayni"] = float(np.mean([plan[i] == E1[i] for i in np.where(cons)[0]])) if cons.any() else float("nan")
    R["uzlasi_plan_en_az_bir_ortak"] = float(np.mean([bool(plan[i] & E1[i]) for i in np.where(cons)[0]])) if cons.any() else float("nan")
    # çapraz denetim ile ilişki
    for nm, m in [("denetim_gecti", aud), ("denetim_kaldi", ~aud)]:
        ii = np.where(m)[0]
        R[f"{nm}_n"] = int(len(ii))
        for k_, E in (("E1", E1), ("E2", E2)):
            R[f"{nm}_plan_{k_}_ayni"] = float(np.mean([plan[i] == E[i] for i in ii]))
            R[f"{nm}_plan_{k_}_en_az_bir_ortak"] = float(np.mean([bool(plan[i] & E[i]) for i in ii]))
    # yöntemlerin yeniden puanlanması
    H = pickle.load(open(os.path.join(OUT, "sentetik_hits.pkl"), "rb"))
    pos = {v: k for k, v in enumerate(IDS)}; ix = [pos[i] for i in ids]
    M = [("Temel: TF-IDF", "en/stem/ad", "TF-IDF (gövde)"), ("USE", "en/plain/ad", "USE"), ("SBERT-mpnet", "en/plain/ad", "SBERT-mpnet"),
         ("Qwen3-Emb-4B", "en/plain/ad+tanim", "Qwen3-Embedding-4B + tanım")] + \
        [(f"BDM {m}", "sifir_atis/en", LLM_NAME[m]) for m in LLM_IDS] + [(f"BDM {m}", "sifir_atis/tr", LLM_NAME[m] + " (Türkçe)") for m in LLM_IDS]
    EN = lambda S_: {TR2EN[x] for x in S_}
    out = []
    for name, cond, lab in M:
        t1 = [H[(name, cond)]["top1"][i] for i in ix]
        hp = np.array([t in EN(s) for t, s in zip(t1, plan)], float)
        h1 = np.array([t in EN(s) for t, s in zip(t1, E1)], float); h2 = np.array([t in EN(s) for t, s in zip(t1, E2)], float)
        hu = np.array([t in EN(x | y) for t, x, y in zip(t1, E1, E2)], float)
        hb = np.array([t in EN(x & y) for t, x, y in zip(t1, E1, E2)], float)
        r = {"yontem": lab, "plan": hp.mean(), "E1": h1.mean(), "E2": h2.mean(), "en_az_bir_etiketleyici": hu.mean(),
             "uzlasi_kumesi": hp[cons].mean() if cons.any() else np.nan, "uzlasi_kumesi_insan": h1[cons].mean() if cons.any() else np.nan}
        err = hp == 0
        r["plan_hatasi_n"] = int(err.sum())
        r["hata_iki_etiketleyici_destekliyor"] = float(hb[err].mean()) if err.any() else np.nan
        r["hata_bir_etiketleyici_destekliyor"] = float(((hu == 1) & (hb == 0))[err].mean()) if err.any() else np.nan
        r["hata_etiketleyici_desteklemiyor"] = float((hu[err] == 0).mean()) if err.any() else np.nan
        out.append(r)
    T = pd.DataFrame(out)
    from scipy.stats import spearmanr
    R["siralama_spearman_plan_E1"] = float(spearmanr(T.plan, T.E1).correlation)
    R["siralama_spearman_plan_E2"] = float(spearmanr(T.plan, T.E2).correlation)
    T.to_csv(os.path.join(OUT, "insan_degerlendirme_yontemler.csv"), index=False, encoding="utf-8-sig")
    det = pd.DataFrame({"sira": [r["sira"] for r in rows], "id": ids, "denetim_gpt5mini": aud,
                        "plan": [";".join(sorted(x)) for x in plan], "E1": [";".join(sorted(x)) for x in E1], "E2": [";".join(sorted(x)) for x in E2],
                        "E1_karar": a.evet, "E2_karar": b.evet, "E1_bicim": a.bicim, "E2_bicim": b.bicim,
                        "E1_listede_olmayan": a.listede_olmayan, "E2_listede_olmayan": b.listede_olmayan})
    det.to_csv(os.path.join(OUT, "insan_degerlendirme_etiketler.csv"), index=False, encoding="utf-8-sig")
    jdump(R, "insan_degerlendirme_ozet.json")
    pd.set_option("display.width", 250)
    print(json.dumps(R, indent=1, ensure_ascii=False, default=float))
    print(T.round(3).to_string(index=False))
    return R, T

if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "olustur":
        olustur(sys.argv[2])
    elif len(sys.argv) >= 4 and sys.argv[1] == "puanla":
        puanla(sys.argv[2], sys.argv[3])
    else:
        sys.exit(__doc__)
