# Afet sonrası ihtiyaç tespiti için doğal dil işleme: kod, simüle veri seti ve sonuçlar

Bu depoda değerlendirme aşamasındaki bir makalenin kodu, simüle veri seti ve sonuçları bulunur. Makalenin künyesi yayımlandığında eklenecektir.

[English summary](#english-summary)

## Özet

- **Veri:**
  - 2.000 simüle sosyal medya paylaşımı (Türkçe ve İngilizce sürüm). Paylaşımlar İstanbul'un 39 ilçesine dağıtılmıştır (ilçe başına 51-52 paylaşım). Her biri, 66 maddelik ihtiyaç listesinden bir ila üç gerçek ihtiyaçla etiketlenmiştir. Veri etiket öncelikli bir protokolle Claude Opus 5.5 ile üretilmiştir: önce ilçe, ihtiyaçlar ve anlatım özellikleri planlanmış, ardından metin yazdırılmıştır. Kayıtların plana uygunluğu bağımsız bir modelle (Claude Sonnet 5.5) denetlenmiştir.
  - Aynı planlardan 1.014'ünün Gemini 3.8 Flash ve GPT-5 mini ile yeniden yazdırıldığı alt küme (üç üreticinin ortak 617 planı karşılaştırmada kullanılmıştır).
  - Gerçek paylaşımlar: HumAID veri setinin deprem bölümünden 225 İngilizce istek tweeti ve 6 Şubat 2023 depremlerine ait bir Kaggle derleminden 423 Türkçe tweet. İki etiketleyicinin birbirinden bağımsız etiketlediği ve tam uyuştuğu 610 paylaşım altın küme olarak kullanılmıştır. Tweet metinleri paylaşılmaz; yalnız kimlikler ve etiketler depodadır.
- **Yöntemler:**
  - 18 dil temsil modeli: benzerlik için ince ayarlanmış cümle kodlayıcılar (USE, SBERT-mpnet, all-distilroberta-v1, ALBERT-paraphrase), güncel çok dilli kodlayıcılar (multilingual-E5-large-instruct, BGE-M3, Qwen3-Embedding 0,6B ve 4B), Türkçe metin için çok dilli SBERT ve BERTurk-STS; ince ayar ablasyonu için ham BERT, RoBERTa ve ALBERT ile NLI ile ince ayarlanmış BERT ve RoBERTa; statik kelime vektörleri GloVe, Word2Vec ve FastText. Karar kuralı, paylaşım ile ihtiyaç adı (ya da ad ve kısa tanım) arasındaki en yüksek kosinüs benzerliğidir.
  - 3 temel çizgi: TF-IDF, rastgele ve çoğunluk.
  - Sıfır atış protokolüyle 5 büyük dil modeli (BDM): Claude Opus 5.5, Claude Sonnet 5.5, Claude Haiku 5.5, GPT-5 mini ve Gemini 3.8 Flash; İngilizce ve özgün Türkçe metinde, tek ve çok etiketli, üç tekrarlı.
  - Az örnekli öğrenme (SetFit ve 66 örnekli BDM istemi), TF-IDF benzerliği düşük paylaşımları BDM'ye yönlendiren kademeli tasarım, ilçe ve konum çıkarımı (alt dize, bulanık sözlük, Türkçe NER).
- **Ölçütler:**
  - Mesaj düzeyi hit@1 ve hit@3, yakın anlam grubu (32 grup) ve sektör (13 sektör) düzeyinde isabet, kısmi puan, çok etiketli mikro ve makro F1, ilçe doğruluğu ve ilçe başına paylaşım sayısına duyarlılığı.
  - %95 önyükleme güven aralıkları, gerçek veride kesin binom aralıkları; McNemar tam testi (Holm düzeltmeli); kendi ailesini kayırma için kümelenmiş standart hatalı doğrusal olasılık ve lojistik model.
  - Açık / karma / örtük anlatım ayrımı; gecikme ve maliyet.
- **Temel bulgular:**
  - Simüle veride (N = 2.000) ihtiyacın adıyla anıldığı açık paylaşımlarda TF-IDF (0,97) BDM'lerle (0,96-0,98) aynı isabete ulaşır. İhtiyacın adlandırılmadığı örtük paylaşımlarda sözcüksel ve gömme yöntemleri 0,11-0,45'e düşer, BDM'ler 0,70-0,88'de kalır.
  - İhtiyaç adlarına kısa tanım eklemek en iyi kodlayıcıyı (Qwen3-Embedding-4B) 0,81'e çıkarır; BDM'lerin genel isabeti 0,88-0,94'tür.
  - BDM üstünlüğü metni yazan modelin ailesine değil, metnin ne kadar örtük olduğuna bağlıdır; değerlendiricinin üreticiyle aynı aileden olması isabeti artırmaz.
  - Gerçek paylaşımlarda BDM'ler 0,93-1,00 isabete ulaşır ve yöntem sıralaması korunur (Spearman ρ İngilizcede 0,94, Türkçede 0,96).
  - Kademeli tasarım paylaşımların yaklaşık üçte birini BDM'ye göndererek BDM'nin isabetine 1 puan yaklaşır, yaklaşık yarısını göndererek ona ulaşır.

## Depo yapısı

```
notebooks/
  NLP_Afet_TekParca.ipynb        Ana defter (Google Colab): veri üretimi ve denetimi, ön işleme,
                                 gömme modelleri ve temel çizgiler, BDM değerlendirmesi,
                                 istatistiksel testler, tablolar, şekiller ve haritalar
  NLP_Afet_BDM_Tamamlama.ipynb   Ek defter: ana çalıştırmada eksik kalan BDM değerlendirmelerini
                                 (GPT-5 mini, Gemini 3.8 Flash) aynı protokolle tamamlar
  NLP_Afet_Revizyon.ipynb        Revizyon defteri: alternatif üreticiler, gerçek paylaşımlar, Türkçe BDM
                                 değerlendirmesi, güncel kodlayıcılar, ihtiyaç tanımları, az örnekli öğrenme,
                                 çok etiketli ve tekrarlı BDM çalıştırmaları, ilçe ve konum çıkarımı
runs/
  20261016_1116/                 Makalede raporlanan ana çalıştırma
    data/                        Veri seti (CSV, XLSX, JSONL) ve ilçe sınırları (geo/)
    results/                     Özet tablolar (CSV, XLSX, MD), ihtiyaç listesi, kalite kontrol sonuçları
      tahminler/                 Her model ve metin türü için paylaşım (mesaj_*) ve ilçe (ilce_*) düzeyi tahminler
    figures/                     Veri dağılımı şekilleri
    maps/                        İlçe düzeyinde ihtiyaç haritaları (HTML ve PNG)
    logs/                        Ayarlar, üretim ve değerlendirme istemleri, kalite kontrol özeti,
                                 ilçe profilleri, model bilgisi, ortam bilgisi, çalıştırma günlüğü
  20261016_0743_claude_maliyet/  Claude modellerinin gerçek API çağrılarıyla ölçülen maliyetleri (aşağıdaki nota bakınız)
  revizyon/                      Revizyon defterinin çalıştırması
    data/                        Gerçek paylaşımların kimlikleri ve etiketleri (metinler hariç), etiketleyici uyumu
    results/                     Veri özetleri ve ihtiyaç listesi
      revizyon/                  Revizyon deneylerinin özet tabloları, alternatif üreticilerin metinleri ve denetimi,
                                 ihtiyaç tanımları, sektör şeması, BDM istemleri, API kullanımı ve maliyeti
      tahminler/                 Paylaşım düzeyi tahminler ve benzerlik puanları (skor_*), mesaj ve ilçe düzeyi sonuçlar
      analiz/                    analiz/revizyon_*.py betiklerinin ürettiği türetilmiş tablolar (makaledeki sayılar)
    figures/                     Veri dağılımı şekilleri
    logs/                        Ayarlar, istemler ve çalıştırma günlüğü
analiz/                          Makaledeki türetilmiş sayıları ve şekilleri depodaki dosyalardan yeniden üreten betikler
                                 (revizyon_*.py: revizyon analizleri)
makale_sekilleri/                Makalenin şekilleri (fig_*.png) ve grafiksel özetler (grafiksel_ozet_*.png)
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

### Alternatif üreticiler

`runs/revizyon/results/revizyon/alt_claude_veri.csv`, `alt_gemini_veri.csv` ve `alt_gpt_veri.csv` dosyaları aynı planların Claude Opus 5.5, Gemini 3.8 Flash ve GPT-5 mini tarafından yazılmış sürümlerini içerir (sütunlar ana veri setindekilerle aynıdır; `uretici` metni yazan modeli, `qc_ok` farklı aileden bir modelin çapraz denetim sonucunu gösterir). Denetim özeti `alternatif_uretici_denetim.csv` dosyasındadır.

### Gerçek paylaşımlar

Dosya: `runs/revizyon/data/gercek_veri_altin_kimlikler.csv`. Her satır bir paylaşımdır; tweet metinleri platform kullanım koşulları nedeniyle depoda yoktur ve kimliklerden yeniden toplanabilir.

| Sütun | Açıklama |
|---|---|
| `kume` | `EN`: HumAID deprem bölümü (istekler ya da acil ihtiyaçlar sınıfı), `TR`: 6 Şubat 2023 depremlerine ait Kaggle tweet derlemi |
| `sira`, `tweet_id` | Satır kimliği ve tweet kimliği |
| `somut_ihtiyac` | Paylaşım belirli bir madde, hizmet ya da ekip istiyor mu (`Evet` / `Hayır`) |
| `ihtiyaclar_tr` | Listedeki ihtiyaçlar (`;` ile ayrılmış, Türkçe adlar) |
| `listede_olmayan`, `listede_olmayan_sektor` | Listede karşılığı olmayan ihtiyaçlar ve sektörleri |
| `anlatim` | Anlatım biçimi (`Açık` / `Örtük`) |
| `il`, `ilce` | Türkçe paylaşımlarda metinde geçen il ve ilçe |

Paylaşımlar iki etiketleyici tarafından birbirinden bağımsız olarak etiketlenmiş; altın kümeye yalnız iki etiketlemenin tam uyuştuğu paylaşımlar alınmıştır (İngilizce 207/225, Türkçe 403/423). Uyum istatistikleri `runs/revizyon/data/gercek_veri_uyum.json` dosyasındadır.

## Yeniden üretim

### 1. Makaledeki sayıları ve şekilleri yeniden üretmek (API ve model indirmesi gerekmez)

```bash
git clone https://github.com/kurbankotan/disaster-needs-assessment-nlp.git
cd disaster-needs-assessment-nlp
pip install -r requirements.txt
python analiz/calistir.py
```

Betik `runs/` altındaki tablo ve tahmin dosyalarını okur. Dosya adlarında Windows'ta geçersiz olan `:` karakteri yerine `_` kullanılmıştır (örneğin `Temel_ TF-IDF`); ana defter Colab'da bu dosyaları `Temel: TF-IDF` adıyla yazar ve analiz betikleri iki biçimi de tanır. Temel sayıları ekrana basar ve şekilleri Liberation Serif yazı tipiyle `makale_sekilleri_yeniden/` klasörüne yazar (yazı tipi sistemde yoksa matplotlib başka bir yazı tipine geçer).

Şekillerin Carlito yazı tipli düzeni ve grafiksel özetler için:

```bash
python analiz/sekiller_carlito.py
```

Bu betik aynı dosyaları okur; genel çerçeve, hit@1, genel karşılaştırma ve anlatım biçimi şekillerini, ayrı paneller hâlinde dört ilçe haritasını ve İngilizce/Türkçe grafiksel özetleri `makale_sekilleri_carlito_yeniden/` klasörüne yazar, haritaların ilçe doğruluklarını da ekrana basar. Carlito, Calibri ile aynı ölçülere sahip açık lisanslı bir yazı tipidir; kurulu değilse matplotlib başka bir yazı tipine geçer (bu durumda yalnızca görünüm değişir).

Betikler Python 3.11 ile `requirements.txt` içindeki sürümlerle sınanmıştır.

### 2. Revizyon analizlerini yeniden üretmek (API ve model indirmesi gerekmez)

```bash
pip install -r requirements-revizyon.txt
python analiz/revizyon_calistir.py
python analiz/revizyon_sekilleri.py
```

`revizyon_calistir.py` analiz betiklerini sırayla çalıştırır (persona ve ilçe duyarlılığı, sentetik veride tüm yöntemler, eşleştirilmiş testler, kademeli tasarım, üretici etkisi, gerçek paylaşımlar) ve türetilmiş tabloları `runs/revizyon/results/analiz/` klasörüne yazar. `revizyon_sekilleri.py` makalenin şekillerini ve grafiksel özetleri Carlito yazı tipiyle `makale_sekilleri_revizyon_yeniden/` klasörüne yazar. Betikler Python 3.13 ile `requirements-revizyon.txt` içindeki sürümlerle sınanmıştır ve depodaki tabloları ve şekilleri birebir yeniden üretir.

Gerçek Türkçe paylaşımlarda TF-IDF metin gerektirdiğinden, metinler depoda olmadığında bu yöntemin kayıtlı paylaşım düzeyi tahminleri (`results/analiz/tahmin_gercek_tr_tfidf_tr_*.csv`) kullanılır. Tweet metinleri yeniden toplanıp aynı sütunlara ek olarak `metin` sütunuyla `runs/revizyon/data/gercek_veri_altin.csv` adıyla konduğunda bu tahminler yeniden hesaplanır.

### 3. Deneyi baştan çalıştırmak (Google Colab)

1. `notebooks/NLP_Afet_TekParca.ipynb` defterini Colab'da açın. Çalışma zamanı olarak GPU ve yüksek RAM seçin; makaledeki çalıştırma NVIDIA A100 (80 GB) ile yapılmıştır (`runs/20261016_1116/logs/ortam.txt`). FastText modeli tek başına yaklaşık 7 GB bellek ister.
2. Colab'ın **Anahtarlar (Secrets)** bölümüne aşağıdaki anahtarları ekleyin:
   - `ANTHROPIC_API_KEY`: veri üretimi, denetim ve Claude değerlendirmesi için gereklidir.
   - `OPENAI_API_KEY`, `GOOGLE_API_KEY`: isteğe bağlıdır; GPT-5 mini ve Gemini değerlendirmesi için kullanılır.

   Anahtarlar bu bölümden, orada yoksa ortam değişkenlerinden okunur. Ayar ve günlük dosyalarına yazılmazlar; günlükte yalnızca her anahtarın tanımlı olup olmadığı yer alır.
3. Tüm ayarlar defterin ilk kod hücresindedir. Drive'da bulunmayan önceden eğitilmiş vektörleri defter özgün kaynaklarından indirir: GloVe glove.840B.300d, Word2Vec GoogleNews-vectors-negative300, FastText cc.en.300.bin ve Universal Sentence Encoder v4. Çıktılar Drive'da `MyDrive/NLP_Disaster/runs/<etiket>/` altında bu depodaki yapıyla oluşturulur.
4. Ana çalıştırmada eksik kalan BDM değerlendirmeleri `notebooks/NLP_Afet_BDM_Tamamlama.ipynb` ile tamamlanır. Ek defter tamamlanan modelleri aynı çalıştırma klasöründeki tablolara ve tahmin dosyalarına ekler.
5. Revizyon deneyleri `notebooks/NLP_Afet_Revizyon.ipynb` ile çalıştırılır. Defter yeni sentetik veri üretmez; ana çalıştırmanın veri setini ve ayarlarını kullanır, çıktıları `MyDrive/NLP_Disaster/runs/revizyon/` altına yazar. Gerçek paylaşım deneyleri için metinli altın etiket dosyası (`gercek_veri_altin.csv`) `MyDrive/NLP_Disaster/gercek_veri/` klasörüne konmalıdır; dosya yoksa bu adımlar atlanır. Makaledeki revizyon çalıştırması Python 3.13 ve NVIDIA A100 (40 GB) ile yapılmıştır.
6. `SMOKE_TEST = True` ayarıyla akış, API ve büyük modeller kullanılmadan sahte veriyle birkaç dakikada denenebilir.

## Notlar

- **BDM yanıtları belirlenimci değildir.** Aynı ayarlarla yeniden çalıştırma farklı yanıtlar üretebilir. Makaledeki sonuçlar `results/tahminler/` altındaki kayıtlı yanıtlara dayanır.
- **GPT-5 mini:** 2.000 paylaşımın 210'u ilk denemede `reasoning_effort` ayarı verilmeden, yani modelin varsayılan düşünme düzeyiyle yanıtlanmıştır. Diğer paylaşımlar `minimal` ayarıyla yanıtlanmıştır. Revizyon defteri bu 210 paylaşımı `minimal` ayarıyla yeniden istemiştir; makalenin revize sürümündeki GPT-5 mini sonuçları protokolle tutarlı bu yanıtlarla hesaplanmıştır (`runs/revizyon/results/tahminler/*gpt-5-mini_protokol_tutarli.csv`). İlk kayıtlar `runs/20261016_1116/` altında olduğu gibi durmaktadır.
- **Gemini:** Ana çalıştırmanın ayar kaydında (`logs/config.json`) Google modeli olarak `gemini-2.5-flash` görünür. Bu model çalıştırma sırasında yeni kullanıcılara kapalı olduğundan API 404 yanıtı vermiştir. Bu nedenle Google değerlendirmesi ek defterle `gemini-3.8-flash` üzerinde yapılmıştır (`logs/bdm_ek_20261016_1252.json`). Gemini 3.8 Flash için değerlendirme sırasında ölçülen gecikme, istemci tarafındaki hız sınırı beklemesini içerdiği için makalede raporlanmamıştır.
- **Claude maliyetleri:** `20261016_1116` çalıştırması Claude yanıtlarını önceki `20261016_0743` çalıştırmasının önbelleğinden almıştır; bu yüzden bu çalıştırmanın tablolarında Claude maliyeti 0 görünür. Gerçek çağrıların maliyeti `runs/20261016_0743_claude_maliyet/` altındadır ve analiz betikleri bu dosyayı kullanır.
- **Şekiller ve haritalar:** `runs/*/figures/` klasörlerinde yalnızca veri dağılımı şekilleri vardır. `makale_sekilleri/` klasöründe makalenin revize sürümünün şekilleri bulunur: `fig_framework`, `fig_hit1`, `fig_all`, `fig_explicit`, `fig_generators`, `fig_real_posts`, `fig_maps` ve `fig_cascade` sırasıyla Şekil 1-8'dir; grafiksel özetler `grafiksel_ozet_tr.png` ve `grafiksel_ozet_en.png` dosyalarındadır. Bu şekiller `analiz/revizyon_sekilleri.py` ile üretilir. Şekil 7'deki dört ilçe haritası `analiz/sekiller_carlito.py` ile ayrı paneller hâlinde üretilir; `fig_maps.png` aynı haritaların birleşik sürümüdür. İlk sürümün şekilleri `analiz/calistir.py` ve `analiz/sekiller_carlito.py` ile yeniden üretilebilir; `fig_cosine.png` ve `fig_transformer.png` ilk sürüme aittir. Ek defterle tamamlanan BDM'lerin (GPT-5 mini, Gemini 3.8 Flash) ilçe haritaları `maps/` klasöründe yoktur; bu modellerin ilçe düzeyi sonuçları `results/tahminler/ilce_sentetik_BDM_*.csv` dosyalarındadır.
- **Model ağırlıkları** depoya konmamıştır. Kaynakları `logs/model_bilgisi.json` dosyasında listelenmiştir.
- **Kademeli tasarımın gecikmesi:** İlk kademedeki TF-IDF dönüşümünün süresi donanıma bağlıdır. Sonuçların birebir yeniden üretilebilmesi için `analiz/revizyon_kademeli.py` analiz makinesinde ölçülen değeri (0,344 ms/paylaşım) kullanır ve çalıştığı makinedeki ölçümü yalnızca bilgi olarak yazdırır. BDM gecikmeleri revizyon çalıştırmasında istemci tarafındaki bekleme hariç API süresinden hesaplanmıştır.
- **Revizyon çalıştırmasının ayar kaydı:** `runs/revizyon/logs/config.json` dosyasından gizli bilgi içeren bir alan çıkarılmıştır; diğer ayarlar olduğu gibidir. Revizyon defteri adında KEY, TOKEN, SECRET ya da PASSWORD geçen hiçbir değişkeni ayar kaydına yazmaz.

## Lisans

- Kod (`notebooks/`, `analiz/`): MIT Lisansı (`LICENSE`).
- Veri seti, sonuçlar ve şekiller (`runs/`, `makale_sekilleri/`): [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- Gerçek paylaşımlar (`runs/revizyon/data/`): yalnız tweet kimlikleri ve bu çalışmada üretilen etiketler paylaşılmıştır. Tweetlerin kendisi ve kaynak veri setleri (HumAID, Kaggle derlemi) kendi lisans ve kullanım koşullarına tabidir.
- İlçe sınırları (`runs/20261016_1116/data/geo/`): geoBoundaries gbOpen TUR ADM2 katmanından alınmıştır (OpenStreetMap kaynaklı). Kullanırken kaynak belirtilmelidir: © OpenStreetMap katkıcıları, geoBoundaries.

## Atıf

Makalenin künyesi yayımlandığında buraya ve `CITATION.cff` dosyasına eklenecektir. O zamana kadar depoya `CITATION.cff` dosyasındaki bilgilerle atıf yapılabilir.

---

## English summary

This repository contains the code, the simulated dataset and the results of a paper that is currently under review (the paper is written in Turkish). The full reference will be added after publication.

- **Data:**
  - 2,000 simulated Turkish-English social media posts, spread over the 39 districts of Istanbul (51-52 posts per district). Each post is labelled with one to three true needs from a 66-item need list. The posts were generated with Claude Opus 5.5 under a label-first protocol and audited by an independent model (Claude Sonnet 5.5).
  - A subset in which 1,014 of the same plans were rewritten by Gemini 3.8 Flash and GPT-5 mini (617 plans common to all three generators are compared).
  - Real posts: 225 English request tweets from the earthquake portion of HumAID and 423 Turkish tweets from a Kaggle corpus of the 6 February 2023 earthquakes, labelled independently by two annotators; the 610 posts on which both labellings fully agree form the gold set. Only tweet IDs and labels are shared, not the tweet texts.
- **Methods:**
  - 18 language representation models, including recent multilingual encoders (multilingual-E5-large-instruct, BGE-M3, Qwen3-Embedding 0.6B and 4B) and a fine-tuning ablation, compared through cosine similarity between each post and the need names (optionally with short need descriptions).
  - 3 baselines: TF-IDF, random and majority.
  - 5 large language models in a zero-shot protocol on English and original Turkish text, with multi-label and repeated runs.
  - Few-shot learning (SetFit, 66-shot LLM prompt), a cascade that routes posts with low TF-IDF similarity to an LLM, and district extraction (substring, fuzzy gazetteer, Turkish NER).
- **Main results:**
  - On explicitly phrased posts TF-IDF (0.97) matches the LLMs (0.96-0.98); on implicitly phrased posts lexical and embedding methods fall to 0.11-0.45, while the LLMs stay at 0.70-0.88.
  - Need descriptions raise the best encoder (Qwen3-Embedding-4B) to 0.81; the LLMs reach 0.88-0.94 overall.
  - The LLM advantage depends on how implicit the texts are, not on which model family wrote them; evaluators show no same-family advantage.
  - On real posts the LLMs reach 0.93-1.00 and the method ranking is preserved (Spearman 0.94 in English, 0.96 in Turkish).

**Quick start:**
- `pip install -r requirements.txt && python analiz/calistir.py` regenerates the figures and key numbers of the first version from the stored outputs. It needs no API keys or model downloads.
- `pip install -r requirements-revizyon.txt && python analiz/revizyon_calistir.py && python analiz/revizyon_sekilleri.py` regenerates all derived tables of the revised paper (`runs/revizyon/results/analiz/`) and its figures.
- To rerun the full experiment, open `notebooks/NLP_Afet_TekParca.ipynb` in Google Colab with a GPU, high RAM and an `ANTHROPIC_API_KEY` secret. `OPENAI_API_KEY` and `GOOGLE_API_KEY` are optional. The additional experiments of the revised paper are in `notebooks/NLP_Afet_Revizyon.ipynb`. LLM outputs are not deterministic; the reported results are based on the stored responses under `runs/`.

**License:** code under MIT; data, results and figures under CC BY 4.0. District boundaries come from geoBoundaries (OpenStreetMap contributors). For the real posts only tweet IDs and our labels are shared; the tweets and source datasets remain under their own terms.
