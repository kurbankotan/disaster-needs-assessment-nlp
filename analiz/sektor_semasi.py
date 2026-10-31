# -*- coding: utf-8 -*-
"""Madde -> sektör şeması (IASC küme yaklaşımına dayalı üst düzey). 66 madde sabittir; sektör düzeyi
listede olmayan ihtiyaçların da değerlendirilebilmesi ve hiyerarşik (kısmi) puanlama için kullanılır."""

SECTORS = [
    # (kod, Türkçe ad, İngilizce ad, kısa tanım)
    ("S01", "Arama-kurtarma", "Search and rescue", "Enkaz altındakileri bulma, çıkarma ve tahliye; bunun için ekip ve ekipman"),
    ("S02", "Sağlık", "Health", "Tıbbi bakım, ilaç, tıbbi malzeme, kan, ruh sağlığı ve psikososyal destek"),
    ("S03", "Gıda ve beslenme", "Food and nutrition", "Gıda, hazır yemek ve bebek beslenmesi"),
    ("S04", "Su, sanitasyon ve hijyen", "Water, sanitation and hygiene", "İçme suyu, tuvalet, kişisel hijyen ve temizlik"),
    ("S05", "Barınma ve temel eşya", "Shelter and non-food items", "Barınak, ısınma, giyim, uyku, mutfak ve aydınlatma eşyası"),
    ("S06", "Enerji ve iletişim", "Energy and telecommunications", "Elektrik, şarj, haberleşme ve internet"),
    ("S07", "Koruma ve güvenlik", "Protection and security", "Güvenlik, çocukların korunması, kayıp yakınların aranması, dil desteği"),
    ("S08", "Lojistik ve ulaşım", "Logistics and transport", "Ulaşım, nakliye, yol erişimi, yardımın taşınması ve dağıtımı"),
    ("S09", "Eğitim", "Education", "Okul, geçici eğitim alanı ve eğitim malzemesi"),
    ("S10", "Kamp yönetimi", "Camp management", "Çadır kent ve toplu yaşam alanlarının yönetimi"),
    ("S11", "Erken iyileştirme ve altyapı", "Early recovery and infrastructure", "Enkaz kaldırma, hasar tespiti, onarım ve yeniden yapım"),
    ("S12", "Nakit ve gönüllü destek", "Cash and volunteers", "Para yardımı ve gönüllü insan gücü"),
    ("S13", "Diğer", "Other", "Yukarıdakilere girmeyen ihtiyaçlar"),
]
SECTOR_NAMES = [s[1] for s in SECTORS]

ITEM_SECTOR = {
    "arama kurtarma ekibi": "Arama-kurtarma", "can yeleği": "Arama-kurtarma",
    "ilaç": "Sağlık", "ağrı kesici": "Sağlık", "antibiyotik": "Sağlık", "soğuk algınlığı ilacı": "Sağlık",
    "ateş düşürücü": "Sağlık", "alerji ilaçları": "Sağlık", "astım ilacı": "Sağlık", "insülin": "Sağlık",
    "kalp ilacı": "Sağlık", "yarabandı": "Sağlık", "bandaj": "Sağlık", "ilk yardım kiti": "Sağlık", "kan": "Sağlık",
    "kan toplama merkezi": "Sağlık", "oksijen cihazı": "Sağlık", "sağlık ekibi": "Sağlık",
    "psikolojik destek ekibi": "Sağlık", "engelli ekipmanı": "Sağlık", "maske": "Sağlık",
    "gıda": "Gıda ve beslenme", "kuru gıda": "Gıda ve beslenme", "konserve": "Gıda ve beslenme",
    "sıcak yemek": "Gıda ve beslenme", "ekmek": "Gıda ve beslenme", "glutesiz yiyecek": "Gıda ve beslenme",
    "tuzsuz-şekersiz yiyecek": "Gıda ve beslenme", "bebek maması": "Gıda ve beslenme", "biberon": "Gıda ve beslenme",
    "içme suyu": "Su, sanitasyon ve hijyen", "sabun": "Su, sanitasyon ve hijyen", "ıslak mendil": "Su, sanitasyon ve hijyen",
    "peçete": "Su, sanitasyon ve hijyen", "diş fırçası": "Su, sanitasyon ve hijyen", "diş macunu": "Su, sanitasyon ve hijyen",
    "hijyenik ped": "Su, sanitasyon ve hijyen", "bebek bezi": "Su, sanitasyon ve hijyen", "seyyar tuvalet": "Su, sanitasyon ve hijyen",
    "temizlik ürünü": "Su, sanitasyon ve hijyen",
    "çadır": "Barınma ve temel eşya", "konteynır": "Barınma ve temel eşya", "battaniye": "Barınma ve temel eşya",
    "uyku tulumu": "Barınma ve temel eşya", "yedek giysi": "Barınma ve temel eşya", "mont": "Barınma ve temel eşya",
    "soba": "Barınma ve temel eşya", "odun": "Barınma ve temel eşya", "elektrikli ısıtıcı": "Barınma ve temel eşya",
    "kap kacak": "Barınma ve temel eşya", "mutfak eşyası": "Barınma ve temel eşya", "fener": "Barınma ve temel eşya",
    "el feneri": "Barınma ve temel eşya", "aydınlatıcı": "Barınma ve temel eşya", "pil": "Barınma ve temel eşya",
    "hırdavat": "Barınma ve temel eşya",
    "jenaratör": "Enerji ve iletişim", "powerbank": "Enerji ve iletişim", "pilli radyo": "Enerji ve iletişim",
    "elektrik saha ekibi": "Enerji ve iletişim",
    "güvenlik güçleri": "Koruma ve güvenlik", "çocuk oyun alanı": "Koruma ve güvenlik", "tercüman": "Koruma ve güvenlik",
    "nakit para": "Nakit ve gönüllü destek", "gönüllü": "Nakit ve gönüllü destek",
    "hayvan maması": "Diğer",
}
assert len(ITEM_SECTOR) == 66 and set(ITEM_SECTOR.values()) <= set(SECTOR_NAMES)
