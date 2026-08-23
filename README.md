# Metro İstanbul Arıza Tespit Sınıflandırıcı

Serbest metinli arıza bildirimlerini (örn. *"Yürüyen merdiven durdu 2. peron"*)
otomatik olarak analiz eden bir NLP sistemi. Tek bir cümleden **üç boyutu
birden** çıkarır — bildirimin amacı, teknik kategorisi ve önceliği — ve
yapısal alanlarla birlikte ilgili bakım ekibine yönlendirir.

Metro İstanbul'da yapılan bir staj kapsamında geliştirilmiştir. Proje
kararlarının, ölçümlerin ve gerekçelerin tam dökümü için [`CLAUDE.md`](CLAUDE.md);
taksonomi ve çıktı sözleşmesi için [`yeni-eklemeler.md`](yeni-eklemeler.md).

## Mimari

```
KULLANICI METNİ
      │
      ▼
┌─────────────────────────────────────────┐
│  BERTurk + LoRA  (tek gövde, üç başlık) │
│    ├── INTENT      5 sınıf              │
│    ├── CATEGORY   11 sınıf              │
│    └── PRIORITY    4 sınıf              │
└─────────────────────────────────────────┘
      │
      ├──► kurallı çıkarım: hat, istasyon, konum, ekipman, belirti, kök sebep
      ├──► P1 kural katmanı (yangın, elektrik çarpması… → koşulsuz P1)
      ├──► gradient × input → evidence
      ├──► eksik bilgi tespiti → kullanıcıya soru
      └──► tekrar tespiti (aynı istasyon + ekipman + 15 dk)
      │
      ▼
  FastAPI (:8000) ──► React arayüz (:5173)
```

**Model:** `dbmdz/bert-base-turkish-cased` + LoRA (PEFT). Üç sınıflandırma
başlığı ortak gövdeyi paylaşır; eğitilen parametre 605K (%0.54), adaptör
**2.3 MB**. Ayrı üç model eğitmek yerine multi-task seçildi — gerekçesi
`src/model.py` modül notunda.

## Kategoriler (11)

| kategori | kapsam (özet) |
| --- | --- |
| Mekanik ve İstasyon | yürüyen merdiven, asansör, turnikenin mekanik arızası, kayar kapılar |
| Elektrik ve Enerji | aydınlatma, jeneratör, katener, üçüncü ray, trafo, pano, sigorta |
| Araç ve Tren | tren kapısı, HVAC, fren/cer, vagon camı ve koltuğu, araç içi anons |
| Sinyalizasyon ve Haberleşme | sinyal arızası, PAKS/PSD, CCTV ve sensörlerin teknik arızası, telsiz |
| Elektronik Sistemler | biletmatik, kart okuyucu, QR, para sıkışması, turnike elektroniği |
| Yol ve Hat | ray kırılması, makas, travers, balast, hat üzerinde cisim |
| İstasyon Güvenliği | güvenlik personeli, kamera görüşünün engellenmesi, yangın/duman |
| Temizlik | çöp, hijyen, kirli zemin, koku, haşere, grafiti |
| Yolcu Hizmetleri | anons içeriği, bilgi ekranları, sefer bilgisi, yönlendirme, yoğunluk |
| Güvenlik ve Asayiş Olayı | saldırı, kavga, hırsızlık, hasta yolcu, şüpheli paket |
| Altyapı ve İnşaat | su sızıntısı, çatlak, tünel yapısı, drenaj, inşaat faaliyeti |

Tam kapsam/istisna metinleri `src/config.py` içindeki `CATEGORIES` sözlüğünde
— tek doğruluk kaynağı orasıdır.

## Boyutlar

**Intent (5):** `fault_report`, `incident_report`, `information_request`,
`complaint`, `suggestion`

**Öncelik (4):** P1 Kritik · P2 Yüksek · P3 Orta · P4 Düşük

P1 için **kural katmanı** modelin önünde çalışır: yangın, elektrik çarpması,
raylara kişi düşmesi, intihar riski, şüpheli paket gibi desenler tahmini ezip
koşulsuz P1 verir. Gerekçe: P1'i kaçırmanın bedeli asimetriktir.

## Gereksinimler

- Python 3.12+ (Apple Silicon'da MPS backend kullanılır, yoksa CPU'ya düşer)
- Node.js 18+ / npm
- macOS, Linux veya Windows (WSL önerilir)

## Kurulum

```bash
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
npm install --prefix frontend
```

Veri üretimi/etiketleme yapacaksanız `.env` gerekir (çalışan sistemi kullanmak
için gerekmez — model ve veri repoda hazır):

```bash
# .env
OPENROUTER_API_KEY=...
GEMINI_API_KEY=...
```

## Çalıştırma

```bash
./execute.sh
```

Backend'i (`:8000`) ve frontend'i (`:5173`) birlikte başlatır. Elle:

```bash
./venv/bin/uvicorn backend.main:app --reload --port 8000
npm run dev --prefix frontend
```

Arayüz: http://localhost:5173 · Swagger: http://localhost:8000/docs

## Proje adımları (uçtan uca yeniden üretmek için)

| adım | komut | çıktı |
| --- | --- | --- |
| 1 — seed üretimi | `python -m src.generate_seed` | `data/seed/seed.jsonl` |
| 2 — kalite triyajı | `python -m src.review` | konsol raporu |
| 2b — çoğaltma | `python -m src.generate_data` | `data/raw/amplified.jsonl` |
| 2c — üç boyutlu etiketleme | `python -m src.relabel` | `data/raw/relabeled.jsonl` |
| 2d — eksik kategori/intent üretimi | `python -m src.generate_missing` | `data/raw/relabeled.jsonl` (ekler) |
| 3 — ön işleme | `python -m src.preprocess` | `data/processed/*.csv` |
| 4a — eğitim (multi-task) | `python -m src.train` | `model/govde/` + `model/basliklar.pt` |
| 4b — değerlendirme | `python -m src.evaluate --hatalari-goster` | `model/degerlendirme.json` |
| 4c — eşik kalibrasyonu | `python -m src.calibrate` | `model/kalibrasyon.json` |
| 4d — öncelik etiket tutarlılığı | `python -m src.oncelik_tutarlilik` | `model/oncelik_tutarlilik.json` |
| 5 — bağımsız set üzerinde test | `python -m src.toplu_test <dosya.jsonl>` | konsol raporu |
| 7 — yapısal çıkarım değerlendirmesi | `python -m src.extract --degerlendir` | `model/extraction_degerlendirme.json` |
| 8 — kategorisiz log kayıtlarını çöz | `python -m src.resolve_logs` | `data/logs.db` güncellenir |

Testler:

```bash
./venv/bin/pytest tests/ -v
```

## API özeti

| yol | ne yapar |
| --- | --- |
| `POST /predict` | metin → intent, kategori, öncelik, yapısal alanlar, evidence, eksik bilgi, tekrar tespiti |
| `GET /health` | servis durumu |
| `GET /model-info` | aktif model, üç görevin boyutları, hiperparametreler, eşikler |
| `GET /categories` | 11 kategori + kapsam/istisna metinleri |
| `GET /intents` | 5 intent + tanımları |
| `GET /priorities` | P1–P4 + kural katmanının tetikleyicileri |
| `GET /examples` | örnek bildirimler (gold setinden, eğitimde hiç kullanılmamış) |
| `POST /logs/verify` | tahmini onayla/düzelt |
| `GET /logs/stats` | log veritabanı özeti |
| `GET /logs/export` | onaylanmış kayıtları JSONL olarak indir |
| `GET /stats/categories` | kategori bazında toplam + canlı sayım |

Örnek istek:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "M4 Kadıköy 2 numaralı girişteki yürüyen merdiven çalışmıyor"}'
```

Örnek yanıttan bir kesit:

```json
{
  "intent": "fault_report",
  "category": "mekanik_istasyon",
  "priority": "P3",
  "line": "M4",
  "station": "Kadıköy",
  "location": "2 numaralı giriş",
  "equipment": "yürüyen merdiven",
  "symptom": "çalışmıyor",
  "root_cause": null,
  "evidence": ["merdiven", "yürüyen"],
  "missing_information": [],
  "possible_duplicate": false,
  "routing_unit": "MEKANIK_ISTASYON"
}
```

## Proje yapısı

```
├── data/
│   ├── seed/           # few-shot yemi + gold test seti
│   ├── raw/            # LLM çoğaltma ve etiketleme çıktıları
│   └── processed/      # train/val/test csv
├── model/              # LoRA gövdesi + üç başlık + değerlendirme raporları
├── src/
│   ├── config.py       # TEK doğruluk kaynağı: taksonomi, boyutlar, eşikler
│   ├── model.py        # çok başlıklı sınıflandırıcı
│   ├── relabel.py      # üç boyutlu toplu etiketleme
│   ├── evidence.py     # gradient × input açıklanabilirlik
│   ├── extract.py      # kurallı yapısal çıkarım
│   └── …               # üretim, ön işleme, eğitim, değerlendirme
├── backend/            # FastAPI servisi
├── frontend/           # React (Vite) arayüzü
├── tests/              # backend entegrasyon testleri
├── CLAUDE.md           # tüm proje geçmişi, kararlar, ölçümler
└── yeni-eklemeler.md   # taksonomi ve çıktı sözleşmesi
```

## Lisans / kullanım

Bu bir staj projesi prototipidir; gerçek üretim verisiyle entegrasyon
yapılmamıştır. Kategori taksonomisi ve veri seti Metro İstanbul'un gerçek
operasyonel yapısını temsil etme iddiasında değildir.
