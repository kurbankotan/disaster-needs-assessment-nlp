# -*- coding: utf-8 -*-
"""Revizyon analizlerinin tamamını sırayla çalıştırır ve türetilmiş tabloları runs/revizyon/results/analiz/ klasörüne yazar.
API anahtarı ya da model indirmesi gerekmez; runs/20261016_1116/ ve runs/revizyon/ altındaki kayıtlı çıktıları okur.

    python analiz/revizyon_calistir.py            # tüm analizler
    python analiz/revizyon_sekilleri.py           # ardından şekiller: makale_sekilleri_revizyon_yeniden/

Sıra önemlidir: revizyon_sentetik.py sonraki betiklerin kullandığı paylaşım düzeyi isabetleri (sentetik_hits.pkl) üretir.
"""
import os, subprocess, sys, time

ANALIZ = os.path.dirname(os.path.abspath(__file__))
ADIMLAR = [
    ("revizyon_persona_duyarlilik.py", "persona, çoklu ihtiyaç (gömme modelleri) ve ilçe duyarlılığı"),
    ("revizyon_sentetik.py", "sentetik veride tüm yöntemler: isabet, hiyerarşi, anlatım biçimi, ilçe doğruluğu"),
    ("revizyon_testler.py", "eşleştirilmiş testler (McNemar, Holm) ve fark güven aralıkları"),
    ("revizyon_kademeli.py", "kademeli tasarım: eğriler, çapraz doğrulama, maliyet ve gecikme"),
    ("revizyon_uretici.py", "üretici modelin etkisi ve kendi ailesini kayırma"),
    ("revizyon_duyarlilik.py", "duyarlılık: çapraz denetimden geçen alt kümeler, esnek puan, çok etiket eşik kuralı, uçtan uca ilçe, persona sayıları"),
    ("revizyon_gercek.py", "gerçek paylaşımlar: kapalı ve açık küme, sıralama ilişkisi"),
]
for betik, aciklama in ADIMLAR:
    print(f"\n=== {betik}: {aciklama}", flush=True)
    t0 = time.time()
    r = subprocess.run([sys.executable, os.path.join(ANALIZ, betik)], cwd=os.path.dirname(ANALIZ))
    if r.returncode != 0:
        sys.exit(f"{betik} hata verdi (çıkış kodu {r.returncode}).")
    print(f"--- {betik} tamam ({time.time() - t0:.0f} sn)")
print("\nTüm revizyon analizleri tamamlandı: runs/revizyon/results/analiz/")
