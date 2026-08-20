# Metro İstanbul Arıza Tespit Sınıflandırıcı

Serbest metinli arıza bildirimlerini (örn. *"Yürüyen merdiven durdu 2. peron"*)
otomatik olarak 8 bakım kategorisinden birine yönlendiren bir NLP sistemi.
Amaç: manuel sınıflandırma yükünü azaltmak, bildirimi doğru bakım ekibine
hızlı yönlendirmek.

Metro İstanbul'da yapılan bir staj kapsamında geliştirilmiştir. Proje
kararlarının, ölçümlerin ve gerekçelerin tam dökümü için [`CLAUDE.md`](CLAUDE.md)
dosyasına bakınız — bu README yalnızca kurulum ve çalıştırma özetidir.

## Mimari

```
seed/gold veri (LLM ile üretildi)
        │
        ▼
  çoğaltma (1600 örnek) ──► ön işleme (train/val/test) ──► BERTurk + LoRA eğitimi
                                                                   │
                                                                   ▼
                                                          FastAPI backend (:8000)
                                                                   │
                                                                   ▼
                                                          React arayüz (:5173)
```

**Model:** `dbmdz/bert-base-turkish-cased` + LoRA (PEFT) — adaptör sadece 2.4 MB.
**Sonuçlar:** gold test setinde accuracy 0.9500, macro F1 0.9497 (hedefler:
0.85 / 0.82). Tam metrik dökümü `model/degerlendirme.json` ve `CLAUDE.md`'de.

## Kategoriler

| kategori | kapsam (özet) |
| --- | --- |
| Araç / Tren | vagon kapısı, fren, klima, makinist kabini |
| İstasyon Mekanik | yürüyen merdiven, asansör, peron kapısı, turnikenin fiziksel arızası |
| Elektrik / Enerji | aydınlatma, jeneratör, katener, trafo, sigorta |
| Yazılım / Sistem / Bilet | bilet otomatı, İstanbulkart yazılımı, PID ekranları, SCADA |
| Güvenlik / Emniyet | CCTV, yangın algılama, acil durum butonu, şüpheli paket |
| Altyapı / İnşaat | su sızıntısı, çatlak, tünel yapısı, drenaj |
| Yolcu / Operasyon | sefer gecikmesi/iptali, anons, peron yoğunluğu, kayıp eşya |
| Temizlik / Çevre | kirlilik, çöp, buzlanma, grafiti |

Tam kapsam/istisna metinleri `src/config.py` içinde `CATEGORIES` sözlüğünde.

## Gereksinimler

- Python 3.12+ (Apple Silicon'da MPS backend kullanılır, yoksa CPU'ya düşer)
- Node.js 18+ / npm
- macOS, Linux veya Windows (WSL önerilir)

## Kurulum

```bash
# 1) Python ortamı
python3 -m venv venv
./venv/bin/pip install -r requirements.txt

# 2) Frontend bağımlılıkları
npm install --prefix frontend

# 3) .env dosyası (yalnızca veri ÜRETİMİ için gerekli — çalışan sistemi
#    kullanmak için gerekmez, model ve veri repoda hazır duruyor)
cp .env.example .env   # yoksa elle oluşturun
# GEMINI_API_KEY / GROQ_API_KEY / OPENROUTER_API_KEY
```

Model ve veri seti repoya dahildir (`model/`, `data/`), yeniden eğitime
gerek yoktur. Doğrudan Çalıştırma bölümüne geçebilirsiniz.

## Çalıştırma

En hızlı yol:

```bash
./execute.sh
```

Bu script backend'i (`:8000`) ve frontend'i (`:5173`) tek komutla ayağa
kaldırır. Ayrıntılı kullanım için `./execute.sh --help`.

Elle çalıştırmak isterseniz:

```bash
# Backend
./venv/bin/uvicorn backend.main:app --reload --port 8000

# Frontend (ayrı terminalde)
npm run dev --prefix frontend
```

Arayüz: http://localhost:5173
API dokümantasyonu (Swagger): http://localhost:8000/docs

## Proje adımları (uçtan uca yeniden üretmek için)

Sırayla çalıştırılması gereken script'ler `src/` altında. Her biri
`CLAUDE.md`'de gerekçesiyle birlikte belgeli.

| adım | komut | çıktı |
| --- | --- | --- |
| 1 — seed/gold üretimi | `python -m src.generate_seed` | `data/seed/*.jsonl` |
| 2 — kalite triyajı | `python -m src.review` | konsol raporu |
| 2b — çoğaltma | `python -m src.generate_data` | `data/raw/amplified.jsonl` |
| 3 — ön işleme | `python -m src.preprocess` | `data/processed/*.csv` |
| 4a — eğitim | `python -m src.train` | `model/adapter_model.safetensors` |
| 4b — değerlendirme | `python -m src.evaluate --kalibrasyon --hatalari-goster` | `model/degerlendirme.json` |
| 4c — eşik kalibrasyonu | `python -m src.calibrate` | `model/kalibrasyon.json` |
| 7 — yapısal çıkarım değerlendirmesi | `python -m src.extract --degerlendir` | `model/extraction_degerlendirme.json` |

Testler:

```bash
./venv/bin/pytest tests/ -v
```

## API özeti

| yol | ne yapar |
| --- | --- |
| `POST /predict` | metin → kategori, güven, olasılık dağılımı, `line`/`station`/`equipment`/`symptom`, `manual_review` |
| `GET /health` | servis durumu |
| `GET /model-info` | aktif model, hiperparametreler, eşikler |
| `GET /categories` | 8 kategori + kapsam metinleri |
| `GET /examples` | örnek bildirimler (gold setinden, eğitimde hiç kullanılmamış) |
| `POST /logs/verify` | tahmini onayla/düzelt (sadece onaylanan kayıtlar ileride elle eğitime katılabilir) |
| `GET /logs/stats` | log veritabanı özeti |
| `GET /logs/export` | onaylanmış kayıtları JSONL olarak indir |
| `GET /stats/categories` | kategori bazında toplam + canlı sayım |

Örnek istek:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Yürüyen merdiven durdu 2. peron"}'
```

## Proje yapısı

```
├── data/
│   ├── seed/          # few-shot yemi + gold test seti (elle gözden geçirilmiş)
│   ├── raw/            # LLM çoğaltma çıktısı
│   └── processed/      # train/val/test/gold_test.csv
├── model/               # eğitilmiş LoRA adaptörü + değerlendirme/kalibrasyon raporları
├── src/                 # veri üretimi, ön işleme, eğitim, değerlendirme, çıkarım
├── backend/              # FastAPI servisi
├── frontend/              # React (Vite) arayüzü
├── tests/                 # backend entegrasyon testleri
├── execute.sh              # backend + frontend'i birlikte başlatır
└── CLAUDE.md                # tüm proje geçmişi, kararlar, ölçümler
```

## Lisans / kullanım

Bu bir staj projesi prototipidir; gerçek üretim verisiyle entegrasyon
yapılmamıştır. Kategori taksonomisi ve veri seti Metro İstanbul'un gerçek
operasyonel yapısını temsil etme iddiasında değildir.
