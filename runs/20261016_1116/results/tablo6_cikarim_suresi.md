| Model | ms_mesaj_toplu | ms_mesaj_tekil | toplu_cihaz | tekil_cihaz | kaynak |
|---|---|---|---|---|---|
| BERT | 0,521 | 46,552 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:bert-base-nli-mean-tokens |
| ALBERT | 0,563 | 45,159 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:albert-base-v2 |
| RoBERTa | 0,517 | 49,310 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:roberta-base-nli-mean-tokens |
| GloVe | 0,043 | 0,064 | CPU | CPU | glove.840B.300d (/content/drive/MyDrive/Colab Notebooks/NLP Modeller/GloVe File/glove.840B.300d-001.txt) |
| Word2Vec | 0,044 | 0,064 | CPU | CPU | word2vec-google-news-300 (/content/drive/MyDrive/Colab Notebooks/NLP Modeller/Word2Vec Files/word2vec-google-news-300.bin) |
| USE | 0,066 | 3,319 | CPU | CPU | universal-sentence-encoder/4 (/content/drive/MyDrive/Colab Notebooks/NLP Modeller/USE Files) |
| FastText | 0,270 | 0,284 | CPU | CPU | cc.en.300.bin (/content/drive/MyDrive/Colab Notebooks/NLP Modeller/FastText Files/cc.en.300.bin) |
| SBERT-mpnet | 0,545 | 46,402 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:all-mpnet-base-v2 |
| BERT-raw | 0,522 | 47,114 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:bert-base-uncased |
| ALBERT-paraphrase | 0,327 | 23,403 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:paraphrase-albert-small-v2 |
| SBERT-çokdilli (TR) | 0,300 | 21,756 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:paraphrase-multilingual-MiniLM-L12-v2 |
| BERTurk-STS (TR) | 0,589 | 46,949 | NVIDIA A100-SXM4-80GB | CPU | sentence-transformers:emrecan/bert-base-turkish-cased-mean-nli-stsb-tr |
