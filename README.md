# Afet sonrası sosyal medya paylaşımlarından ilçe düzeyinde ihtiyaç tespiti

Bu depoda aşağıdaki makalenin kodu, simüle veri seti ve sonuçları bulunur:

> Kotan K., *Afetlerde hızlı risk profillemesi için doğal dil işleme: Bir karar destek çerçevesi*, Gazi Üniversitesi Mühendislik Mimarlık Fakültesi Dergisi (değerlendirme aşamasında).

[English summary](#english-summary)

## Özet

- **Veri:** 2.000 simüle sosyal medya paylaşımı (Türkçe ve İngilizce sürüm). Paylaşımlar İstanbul'un 39 ilçesine dağıtılmıştır (ilçe başına 51-52 paylaşım). Her biri, 66 maddelik ihtiyaç listesinden bir ila üç gerçek ihtiyaçla etiketlenmiştir. Veri etiket öncelikli bir protokolle Claude Opus 5.5 ile üretilmiştir: önce ilçe, ihtiyaçlar ve anlatım özellikleri planlanmış, ardından metin yazdırılmıştır. Kayıtların plana uygunluğu bağımsız bir modelle (Claude Sonnet 5.5) denetlenmiştir.
- **Yöntemler:**
  - 12 dil temsil modeli: BERT, ALBERT, RoBERTa, GloVe, Word2Vec, USE ve FastText; ek olarak SBERT-mpnet, BERT-raw ve ALBERT-paraphrase; Türkçe metin için çok dilli SBERT ve BERTurk-STS. Bu modellerde karar kuralı, paylaşım ile ihtiyaç adları arasındaki en yüksek kosinüs benzerliğidir.
  - 3 temel çizgi: TF-IDF, rastgele ve çoğunluk.
  - Sıfır atış protokolüyle 5 büyük dil modeli (BDM): Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5, GPT-5 mini ve Gemini 3.8 Flash.
- **Ölçütler:**
  - Mesaj düzeyi hit@1 ve hit@3, ilçe doğruluğu.
  - %95 önyükleme güven aralıkları; McNemar tam testi (Holm düzeltmeli).
  - Açık / karma / örtük anlatım ayrımı; gecikme ve maliyet.
- **Temel bulgular** (N = 2.000):
  - BDM'lerin hit@1 değeri 0,88-0,94'tür.
  - En iyi gömme modeli USE'dir (0,71; işlemcide 3,3 ms/paylaşım). Sözcüksel TF-IDF temel çizgisi de 0,71'e ulaşmıştır.
  - İhtiyacın adıyla anılmadığı örtük paylaşımlarda en iyi cümle kodlayıcılar 0,33-0,37'de, TF-IDF 0,11'de, BDM'ler ise 0,70-0,88'de kalmıştır.

## Depo yapısı

```
notebooks/
  NLP_Afet_TekParca.ipynb        Ana defter (Google Colab): veri üretimi ve denetimi, ön işleme,
                                 gömme modelleri ve temel çizgiler, BDM değerlendirmesi,
                                 istatistiksel testler, tablolar, şekiller ve haritalar
  NLP_Afet_BDM_Tamamlama.ipynb   Ek defter: ana çalıştırmada eksik kalan BDM değerlendirmelerini
                                 (GPT-5 mini, Gemini 3.8 Flash) aynı protokolle tamamlar
runs/
  20261016_1116/                 Makalede raporlanan çalıştırma
    data/                        Veri seti (CSV, XLSX, JSONL) ve ilçe sınırları (geo/)
    results/                     Özet tablolar (CSV, XLSX, MD), ihtiyaç listesi, kalite kontrol sonuçları
      tahminler/                 Her model ve metin türü için paylaşım (mesaj_*) ve ilçe (ilce_*) düzeyi tahminler
    figures/                     Veri dağılımı şekilleri
    maps/                        İlçe düzeyinde ihtiyaç haritaları (HTML ve PNG)
    logs/                        Ayarlar, üretim ve değerlendirme istemleri, kalite kontrol özeti,
                                 ilçe profilleri, model bilgisi, ortam bilgisi, çalıştırma günlüğü
  20261016_0743_claude_maliyet/  Claude modellerinin gerçek API çağrılarıyla ölçülen maliyetleri (aşağıdaki nota bakınız)
analiz/                          Makaledeki türetilmiş sayıları ve şekilleri depodaki dosyalardan yeniden üreten betikler
makale_sekilleri/                Makaledeki şekiller (300 dpi PNG)
```

## Veri seti

Dosya: `runs/20261016_1116/data/sentetik_paylasimlar_a8bfe68ee1.csv` (aynı içerik `.xlsx` ve üretim kayıtlarıyla `.jsonl` biçiminde de vardır). Her satır bir paylaşımdır.

| Sütun | Açıklama |
|---|---|
| `id` | Paylaşım kimliği (0-1999) |
| `Message` | Türkçe paylaşım |
| `English` | Aynı paylaşımın İngilizce sürümü |
| `district` | Planlanan ilçe (39 ilçe) |
| `label1`, `label2`, `label3` | Planlanan gerçek ihtiyaçlar (Türkçe, 66 maddelik listeden; bir ila üç) |
| `n_needs` | İhtiyaç sayısı |
| `explicitness` | Anlatım biçimi: `explicit` (açık), `mixed` (karma), `implicit` (örtük) |
| `persona`, `register`, `length` | Planlanan yazar kimliği, kayıt dili ve uzunluk |
| `typos`, `no_diacritics`, `hashtag`, `emoji`, `neighbourhood`, `numbers` | Planlanan biçim özellikleri: yazım hatası, Türkçe karakter eksikliği, etiket, emoji, mahalle adı, sayısal ifade |
| `gen_model`, `gen_round`, `gen_id`, `source` | Üretim modeli, üretim turu, üretim kimliği ve kaynak (`synthetic`) |
| `qc_needs_ok`, `qc_missing`, `qc_extra`, `qc_district_ok`, `qc_ok` | Bağımsız model denetiminin sonuçları: atanan ihtiyaçların metinde bulunup bulunmadığı, eksik ve fazladan ihtiyaçlar, ilçe uygunluğu, genel sonuç |
| `Time`, `Location`, `hour` | Şema uyumu için yapay olarak üretilmiş alanlar. Tarih 2025-01-01 ile 2025-01-04 arasından rastgele seçilmiştir. Konum, ilçe merkezine küçük rastgele sapma eklenerek oluşturulmuştur; saat de rastgeledir. Bu alanlar analizde kullanılmamıştır. |

İhtiyaç listesinin Türkçe adları ve İngilizce karşılıkları `results/ihtiyac_listesi.csv` dosyasındadır. Paylaşımların tamamı simüledir ve gerçek kişilere ait değildir. Üretim istemi (`logs/prompt_uretim.txt`) telefon numarası, gerçek kişi adı, bağlantı ve @ ile kullanıcı anma kullanılmamasını şart koşmuştur.

## Yeniden üretim

### 1. Makaledeki sayıları ve şekilleri yeniden üretmek (API ve model indirmesi gerekmez)

```bash
git clone https://github.com/kurbankotan/disaster-needs-assessment-nlp.git
cd disaster-needs-assessment-nlp
pip install -r requirements.txt
python analiz/calistir.py
```

Betik `runs/` altındaki tablo ve tahmin dosyalarını okur. Dosya adlarında Windows'ta geçersiz olan `:` karakteri yerine `_` kullanılmıştır (örneğin `Temel_ TF-IDF`); ana defter Colab'da bu dosyaları `Temel: TF-IDF` adıyla yazar ve analiz betikleri iki biçimi de tanır. Makaledeki şekilleri `makale_sekilleri_yeniden/` klasörüne yazar ve temel sayıları ekrana basar. Şekillerde Liberation Serif yazı tipi kullanılır; sistemde yoksa matplotlib başka bir yazı tipine geçer. Betikler Python 3.11 ile `requirements.txt` içindeki sürümlerle sınanmıştır.

### 2. Deneyi baştan çalıştırmak (Google Colab)

1. `notebooks/NLP_Afet_TekParca.ipynb` defterini Colab'da açın. Çalışma zamanı olarak GPU ve yüksek RAM seçin; makaledeki çalıştırma NVIDIA A100 (80 GB) ile yapılmıştır (`runs/20261016_1116/logs/ortam.txt`). FastText modeli tek başına yaklaşık 7 GB bellek ister.
2. Colab'ın **Anahtarlar (Secrets)** bölümüne aşağıdaki anahtarları ekleyin:
   - `ANTHROPIC_API_KEY`: veri üretimi, denetim ve Claude değerlendirmesi için gereklidir.
   - `OPENAI_API_KEY`, `GOOGLE_API_KEY`: isteğe bağlıdır; GPT-5 mini ve Gemini değerlendirmesi için kullanılır.

   Anahtarlar bu bölümden, orada yoksa ortam değişkenlerinden okunur. Ayar ve günlük dosyalarına yazılmazlar; günlükte yalnızca her anahtarın tanımlı olup olmadığı yer alır.
3. Tüm ayarlar defterin ilk kod hücresindedir. Drive'da bulunmayan önceden eğitilmiş vektörleri defter özgün kaynaklarından indirir: GloVe glove.840B.300d, Word2Vec GoogleNews-vectors-negative300, FastText cc.en.300.bin ve Universal Sentence Encoder v4. Çıktılar Drive'da `MyDrive/NLP_Disaster/runs/<etiket>/` altında bu depodaki yapıyla oluşturulur.
4. Ana çalıştırmada eksik kalan BDM değerlendirmeleri `notebooks/NLP_Afet_BDM_Tamamlama.ipynb` ile tamamlanır. Ek defter tamamlanan modelleri aynı çalıştırma klasöründeki tablolara ve tahmin dosyalarına ekler.
5. `SMOKE_TEST = True` ayarıyla akış, API ve büyük modeller kullanılmadan sahte veriyle birkaç dakikada denenebilir.

## Notlar

- **BDM yanıtları belirlenimci değildir.** Aynı ayarlarla yeniden çalıştırma farklı yanıtlar üretebilir. Makaledeki sonuçlar `results/tahminler/` altındaki kayıtlı yanıtlara dayanır.
- **GPT-5 mini:** 2.000 paylaşımın 210'u ilk denemede `reasoning_effort` ayarı verilmeden, yani modelin varsayılan düşünme düzeyiyle yanıtlanmıştır. Diğer paylaşımlar `minimal` ayarıyla yanıtlanmıştır. Makaledeki sonuçlar bu kayıtlarla hesaplanmıştır. Ek defterdeki `REDO_IDS` ayarı, bu paylaşımları `minimal` ayarıyla yeniden ister.
- **Gemini:** Ana çalıştırmanın ayar kaydında (`logs/config.json`) Google modeli olarak `gemini-2.5-flash` görünür. Bu model çalıştırma sırasında yeni kullanıcılara kapalı olduğundan API 404 yanıtı vermiştir. Bu nedenle Google değerlendirmesi ek defterle `gemini-3.8-flash` üzerinde yapılmıştır (`logs/bdm_ek_20261016_1252.json`). Gemini 3.8 Flash için değerlendirme sırasında ölçülen gecikme, istemci tarafındaki hız sınırı beklemesini içerdiği için makalede raporlanmamıştır.
- **Claude maliyetleri:** `20261016_1116` çalıştırması Claude yanıtlarını önceki `20261016_0743` çalıştırmasının önbelleğinden almıştır; bu yüzden bu çalıştırmanın tablolarında Claude maliyeti 0 görünür. Gerçek çağrıların maliyeti `runs/20261016_0743_claude_maliyet/` altındadır ve analiz betikleri bu dosyayı kullanır.
- **Şekiller ve haritalar:** `runs/20261016_1116/figures/` klasöründe yalnızca veri dağılımı şekilleri vardır. Model karşılaştırma şekillerinin güncel hâlleri `makale_sekilleri/` klasöründedir. Ek defterle tamamlanan BDM'lerin (GPT-5 mini, Gemini 3.8 Flash) ilçe haritaları `maps/` klasöründe yoktur; bu modellerin ilçe düzeyi sonuçları `results/tahminler/ilce_sentetik_BDM_*.csv` dosyalarındadır.
- **Model ağırlıkları** depoya konmamıştır. Kaynakları `logs/model_bilgisi.json` dosyasında listelenmiştir.

## Lisans

- Kod (`notebooks/`, `analiz/`): MIT Lisansı (`LICENSE`).
- Veri seti, sonuçlar ve şekiller (`runs/`, `makale_sekilleri/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- İlçe sınırları (`runs/20261016_1116/data/geo/`): geoBoundaries gbOpen TUR ADM2 katmanından alınmıştır (OpenStreetMap kaynaklı). Kullanırken kaynak belirtilmelidir: © OpenStreetMap katkıcıları, geoBoundaries.

## Atıf

Bu depoyu kullanırsanız lütfen makaleye atıf yapın. Künye bilgileri `CITATION.cff` dosyasındadır; makale yayımlandığında güncellenecektir.

---

## English summary

This repository contains the code, the simulated dataset and the results of the paper *Natural language processing for rapid risk profiling in disasters: A decision support framework* (K. Kotan, under review at the Journal of the Faculty of Engineering and Architecture of Gazi University).

- **Data:** 2,000 simulated Turkish-English social media posts, spread over the 39 districts of Istanbul (51-52 posts per district). Each post is labelled with one to three true needs from a 66-item need list. The posts were generated with Claude Opus 5.5 under a label-first protocol and audited by an independent model (Claude Sonnet 5.5).
- **Methods:**
  - 12 language representation models, compared through cosine similarity between each post and the need names.
  - 3 baselines: TF-IDF, random and majority.
  - 5 large language models in a zero-shot protocol.
- **Main results:**
  - LLMs reach hit@1 0.88-0.94.
  - The best embedding model is USE (0.71; 3.3 ms per post on CPU), and TF-IDF also reaches 0.71.
  - On implicitly phrased posts, the best sentence encoders fall to 0.33-0.37 and TF-IDF to 0.11, while the LLMs stay at 0.70-0.88.

**Quick start:**
- `pip install -r requirements.txt && python analiz/calistir.py` regenerates the paper figures and key numbers from the stored outputs. It needs no API keys or model downloads.
- To rerun the full experiment, open `notebooks/NLP_Afet_TekParca.ipynb` in Google Colab with a GPU, high RAM and an `ANTHROPIC_API_KEY` secret. `OPENAI_API_KEY` and `GOOGLE_API_KEY` are optional. LLM outputs are not deterministic; the reported results are based on the stored responses in `runs/20261016_1116/results/tahminler/`.

**License:** code under MIT; data, results and figures under CC BY 4.0. District boundaries come from geoBoundaries (OpenStreetMap contributors).
