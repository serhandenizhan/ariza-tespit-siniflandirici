# Metro İstanbul Arıza Tespit Sınıflandırıcı — Proje Bağlamı

Bu dosya, projenin claude.ai sohbetinde geçen tüm geçmişini özetler. Buradaki
her karar, sayı ve gerekçe önceki bir konuşmadan gelir — tahmin veya varsayım
yok. Staj sunumunda kullanılacağı için doğruluğu kritik.

## Kim, Ne, Neden

Metro İstanbul'da staj yapan bir yazılım mühendisliği öğrencisi, gün içinde
gelen serbest metinli arıza bildirimlerini otomatik kategorilere ayıran bir
NLP modeli + web arayüzü geliştiriyor. Amaç: manuel sınıflandırma yükünü
azaltmak, bildirimi doğru bakım ekibine hızlı yönlendirmek.

Proje iki katmanlı: (1) staj süresince çalışan bir prototip, (2) ileride
gerçek verilerle kuruma entegre edilebilecek bir sistem.

## Çalışma Dizini

```
/Users/serhan/Desktop/ariza-tespit-siniflandirici/
```

Ortam: MacBook Air (Apple Silicon, 16GB RAM), Python venv, Ollama kurulu ve
çalışıyor (model: qwen2.5:14b — gemma2:9b ile karşılaştırma planlanmıştı ama
Nemotron sonuçları çok iyi çıktığı için henüz yapılmadı, gerek kalmayabilir).

## Klasör Yapısı (mevcut durum)

```
ariza-tespit-siniflandirici/
├── data/
│   ├── seed/
│   │   ├── seed.jsonl              # few-shot yemi, 93 kayit
│   │   ├── gold.jsonl              # bozulmamis test seti, 80 kayit (8x10)
│   │   ├── seed_v1_backup.jsonl    # Gemini ile ilk deneme (arsiv)
│   │   ├── gold_v1_backup.jsonl
│   │   ├── seed_v2_groq_backup.jsonl
│   │   ├── gold_v2_groq_backup.jsonl
│   │   ├── seed_v3_groq_fixed_backup.jsonl  # Groq/Llama3.3 duzeltilmis prompt
│   │   ├── gold_v3_groq_fixed_backup.jsonl
│   │   └── gold_v4_pre_guvenlik_backup.jsonl  # guvenlik yeniden uretiminden once
│   ├── raw/
│   │   ├── amplified.jsonl         # Adim 2b ciktisi, 1600 kayit (%100 Nemotron)
│   │   └── amplified_ollama_backup.jsonl  # degisim oncesi 1586 kayit (arsiv/kiyas)
│   └── processed/                  # Adim 3 ciktisi
│       ├── clean.csv               # 1600 (bolunmemis, temizlenmis havuz)
│       ├── train.csv               # 1280
│       ├── val.csv                 #  160
│       ├── test.csv                #  160
│       └── gold_test.csv           #   80  (gold.jsonl'den, egitime GIRMEZ)
├── model/                          # Adim 4 ciktisi, 3.0 MB
│   ├── adapter_model.safetensors   # LoRA agirliklari (sadece 2.4 MB!)
│   ├── adapter_config.json
│   ├── tokenizer.json / tokenizer_config.json
│   ├── egitim_ozeti.json           # hiperparametreler + epoch gecmisi
│   └── degerlendirme.json          # test/gold metrikleri
├── src/
│   ├── __init__.py
│   ├── config.py                   # TEK dogruluk kaynagi, asagida detay
│   ├── generate_seed.py            # seed/gold uretimi (coklu saglayici, --category)
│   ├── generate_data.py            # Adim 2b -- cogaltma (hibrit saglayici)
│   ├── preprocess.py               # Adim 3 -- kumeleme + train/val/test bolme
│   ├── review.py                   # kalite triyaji (seed/gold/amplified)
│   ├── apply_review.py             # elle onaylanan duzeltmeleri uygular (idempotent)
│   ├── check_openrouter_models.py  # OpenRouter canli model/fiyat listesi
│   ├── train.py                    # Adim 4a -- BERTurk + LoRA egitimi
│   └── evaluate.py                 # Adim 4b -- iki test seti + confusion matrix
├── backend/
│   └── main.py                      # YOK -- Adim 5'te yazilacak (FastAPI)
├── frontend/                        # YOK -- Adim 6'da yazilacak (React)
├── venv/
├── requirements.txt                 # pip freeze ile donduruldu (35 paket)
└── .env                             # GEMINI_API_KEY, GROQ_API_KEY,
                                      # OPENROUTER_API_KEY var; ANTHROPIC_API_KEY yok
```

NOT: `data/raw`, `data/processed`, `model` bos olduklarinda git'e girmez (git
bos klasor takip etmez), ama `config.py` import edilir edilmez bunlari kendisi
olusturur -- klonlayan icin hicbir sey kirilmaz.

## config.py — Projenin Tek Doğruluk Kaynağı

Tüm diğer modüller (mevcut ve yazılacak olanlar) buradan import eder. Asla
kategori/stil/hiperparametre tanımı başka bir dosyada tekrarlanmaz.

### Kategori Taksonomisi (8 kategori)

**Ayrım ilkesi: kategori, bildirimin hangi BAKIM EKİBİNE yönlendirileceğini
belirtir. Arızanın nesnesi değil, sorumlusu belirleyicidir.**

| key | display | kapsam (özet) |
| --- | --- | --- |
| `arac_tren` | Araç / Tren | Trenin üzerindeki her şey: vagon kapısı, fren, klima, çer/motor, kabin ekipmanı, tekerlek, koltuk, vagon içi aydınlatma/anons, makinist kabini |
| `istasyon_mekanik` | İstasyon Mekanik | Yürüyen merdiven, asansör, peron kapısı (PSD), turnikenin **fiziksel** arızası (kol dönmüyor, kapak takılı, gövde hasarlı), otomatik giriş kapıları, bariyerler |
| `elektrik_enerji` | Elektrik / Enerji | İstasyon aydınlatması, elektrik kesintisi, jeneratör, UPS, elektrik panosu, katener hattı, üçüncü ray, trafo, kablo arızası, sigorta |
| `yazilim_sistem` | Yazılım / Sistem / Bilet | Bilet satış otomatı, İstanbulkart okuyucu **yazılımı**, PID ekranları, sunucu, ağ kesintisi, uygulama donması, veritabanı, SCADA arayüzü |
| `guvenlik_emniyet` | Güvenlik / Emniyet | CCTV, yangın algılama/söndürme, acil durum butonu, acil çıkış, yetkisiz giriş, turnikeden **atlama**, şüpheli paket, tahliye anonsu |
| `altyapi_insaat` | Altyapı / İnşaat | Su sızıntısı, tavan/duvar/zemin hasarı, çatlak, tünel yapısı, drenaj, kanalizasyon, ray hattı yapısal durumu, korkuluk, fayans |
| `yolcu_operasyon` | Yolcu / Operasyon | Sefer gecikmesi/iptali, seferlerin seyreltilmesi, anons yapılmaması, peron yoğunluğu, kayıp eşya, personel eksikliği, tarife sorunu |
| `temizlik_cevre` | Temizlik / Çevre | Kirlilik, çöp birikmesi, koku, tuvalet temizliği, buzlanma, kaygan zemin, haşere, kış şartları (tuzlama), grafiti |

**Kritik sınır örneği (turnike):** aynı ekipman üç farklı kategoriye
düşebilir, kural nettir:

- Fiziksel arıza (kol dönmüyor, kapak kırık) → `istasyon_mekanik`
- Kart okumama / yazılım hatası → `yazilim_sistem`
- Atlama / yetkisiz geçiş → `guvenlik_emniyet`

Her kategorinin `config.py`'de tam `scope` (kapsam) ve `exclude` (hariç
tutulanlar) metni var, LLM prompt'larına birebir enjekte ediliyor.

### Stil Varyantları (4 stil, gerçekçilik için)

Gerçek personel her zaman düzgün yazmaz. Her kategori için 4 stilde örnek
üretiliyor:

| stil | uzunluk | açıklama |
| --- | --- | --- |
| `standart` | 8-18 kelime | Düzgün, kurallı tam cümle |
| `devrik` | 4-9 kelime | Acele yazılmış, kısa, eksiltili (özne/yüklem sırası bozuk olabilir) |
| `yazim_yanlisi` | 5-14 kelime | Türkçe karakter eksikliği, klavye hatası |
| `cok_kisa` | 3-6 kelime | Telgraf tarzı, sadece ekipman + belirti |

**Önemli tasarım kararı:** Türkçe aksan düşürme (`güvenlik`→`guvenlik`,
`Şişli`→`Sisli`) **sadece** `yazim_yanlisi` stiline özgü değil — gerçek
hayatta İngilizce klavyeyle hızlı yazan biri her stilde bunu yapabilir.
Bu davranış bir "hata" değil, modelin dayanıklı olması gereken doğal bir
varyasyon olarak kabul edildi (bkz. review.py bölümü).

### Hedef Veri Hacmi — hedef vs GERÇEKLEŞEN

| set | hedef | gerçekleşen | not |
| --- | --- | --- | --- |
| seed | 12/kat × 8 = 96 | **93** | elle triyajda 3 kayıt silindi |
| gold | 10/kat × 8 = 80 | **80** | tam, 8 kategori × 10 |
| çoğaltma | 200/kat × 8 = 1600 | **1600** | tam; %100 Nemotron. SINIR düzeltmeleri sonrası kategori dengesi 202/200/199 (±2) |

Çoğaltma sonrası bölme (Adım 3): train **1280** / val **160** / test **160**
(%80/%10/%10). Kategori başına train 159-162 arası (SINIR düzeltmeleri
dengeyi ±2 kaydırdı), val ve test her kategoride tam 20. Ayrıca gold_test
**80** (eğitime hiç girmez).

(Not: orijinal PDF taslağında 70/kategori × 6 kategori = 420 yazıyordu.
Kategori sayısı 6'dan 8'e, hedef hacim 70'ten 200'e çıkarıldı — rapor
güncellenmeli.)

### Çoğaltma Ayarları (Adım 2b, config.py)

- `AMPLIFY_PROVIDER = "hybrid"` — OpenRouter birincil, kalıcı hatada Ollama
- `NEAR_DUP_THRESHOLD = 0.85` (üretimde red) / `CLUSTER_THRESHOLD = 0.80`
  (bölmede kümeleme) — iki ayrı eşik, gerekçesi `config.py`'de
- `AMPLIFY_BATCH_SIZE = 40` — çağrı başına örnek (gerekçesi aşağıda)
- `AMPLIFY_FEWSHOT_N = 6`, `AMPLIFY_AVOID_N = 12`
- `OLLAMA_MODEL = "qwen2.5:14b"`, `OLLAMA_NUM_CTX = 8192`,
  `OLLAMA_NUM_PREDICT = 4096`

**`OLLAMA_NUM_CTX` neden açıkça ayarlandı:** Ollama'nın varsayılan bağlam
penceresi 2048 token. Çoğaltma prompt'u (kategori kapsamı + stil tanımı +
few-shot + "bunları tekrarlama" listesi) bunun büyük kısmını yiyor, çıktıya
yer kalmıyordu: 25 örnek istenmesine rağmen model 1-2 örnek döndürüp
kesiliyordu. 8192'ye çıkarılınca aynı iş 5 çağrıda 8 kayıt yerine 7 çağrıda
40 kayıt üretti.

### Eğitim Hiperparametreleri (henüz kullanılmadı, Adım 4 için hazır)

- Model: `dbmdz/bert-base-turkish-cased`
- PEFT/LoRA: r=16, alpha=32, dropout=0.1, target_modules=["query","value"]
- MAX_LENGTH=64, NUM_EPOCHS=12, BATCH_SIZE=16, **LEARNING_RATE=5e-4**
  (2e-5 degildi -- bkz. Adim 4 bolumu, en onemli hata buydu)
- `AUGMENT_ASCII_FOLD = True` — egitimde aksansiz kopyalar da eklenir
  (train 1280 → 2219). Gerekcesi ve olcumu Adim 4 bolumunde.
- Split: %80 train / %10 val / %10 test
- Başarı kriteri: accuracy ≥ 0.85, macro F1 ≥ 0.82, hiçbir sınıf F1 < 0.75

### Servis Ayarları (henüz kullanılmadı, Adım 5 için hazır)

- `CONFIDENCE_THRESHOLD = 0.70` — **kalibre edildi** (19 Ağu). Bu eşiğin
  altında `low_confidence: true` döner. Ölçüm: doğru tahminlerde ortalama
  güven 0.95, yanlışlarda 0.71. 0.60 gold'da 8 hatanın 2'sini yakalıyordu,
  0.70 ise 5'ini — aynı maliyetle (1 boşuna işaret).
- `MARGIN_THRESHOLD = 0.40` — **yeni.** `top1 − top2` bu değerin altındaysa
  `/predict` birincil + ikincil kategori döner. Taksonomi sınır sorunlarına
  kural yazmak yerine getirilen genel çözüm (aşağıda detaylı).

## LLM Sağlayıcı Yolculuğu (önemli — rapora doğrudan girebilecek içerik)

Seed/gold üretimi için 4 farklı sağlayıcı denendi, sonuçlar `review.py` ile
ölçüldü. Bu, projenin metodoloji bölümüne güçlü bir katkı:

| Deneme | Sağlayıcı/Model | Seed işaretli | Gold işaretli | Not |
| --- | --- | --- | --- | --- |
| v1 | Gemini (çeşitli modeller) | %15 | %29 | Kota/model erişim sorunları yüzünden terk edildi |
| v2 | Groq / llama-3.3-70b-versatile | %82 | %56 | Config'teki Türkçe metin ASCII'ydi (kök sebep) |
| v3 | Groq / llama-3.3-70b-versatile (Türkçe düzeltildi) | %70 | %70 | Prompt dili düzelince iyileşti ama model hâlâ çok-kısıtlı talimatlara uyamadı |
| **v4** | **OpenRouter / nvidia/nemotron-3-ultra-550b-a55b:free** | **%14** | **%4** | **Kazanan.** Ücretsiz, kart istemiyor, v1'i bile geçti |

> **Bu sayılar 19 Ağu 2026'da bugünkü `review.py` ile YENİDEN ölçüldü.** Önceki
> sürümde farklı rakamlar yazıyordu (v1 %18/%32, v2 %91/%81, v3 %72/%69,
> v4 %17/%9) çünkü o ölçümlerden sonra araç iki kez değişti: aksan kontrolü
> işaretli sayımından çıkarıldı ve `similarity()` simetrik hâle getirildi.
> Sıralama ve sonuç değişmedi, ama rapordaki her sayının bugünkü araçla
> üretilebilir olması için tablo tazelendi. Ölçüm şu komutla tekrarlanabilir:
> `python -m src.review --file seed` (yedek dosyalar `data/seed/` altında
> duruyor — bu yüzden silinmediler).

**Ders 1 — kaynak kodun kendi Türkçesi önemli:** `config.py`'deki kategori
açıklamaları, kurallar, istasyon adları başlangıçta ASCII yazılmıştı (bir
önceki PDF-font sorunuyla karıştırılıp gereksiz genellenmişti). Bu, LLM
prompt'larına doğrudan enjekte edildiği için modele yanlış stil sinyali
verdi. Düzeltilince (v2→v3) işaretli oran ciddi düştü.

**Ders 2 — model ölçeği ve talimat-takibi kapasitesi asıl belirleyici:**
Aynı düzgün Türkçe prompt'la bile Llama 3.3 70B (v3) çok-kısıtlı talimatlara
(kategori + stil + uzunluk + kod oranı + sızıntı oranı aynı anda) uyamadı.
550B/55B-aktif Nemotron 3 Ultra (v4) aynı prompt'la çok daha iyi sonuç verdi
— talimat-takibi, ham dil kalitesinden çok modelin talimat karmaşıklığını
yönetme kapasitesiyle ilgili.

**Ders 3 — ücretsiz katmanlar güvenilmez, canlı sorgula, tahmin etme:**
Gemini'de sırasıyla günlük kota (20/gün), model erişim kısıtı ("yeni
kullanıcılara kapalı"), ve bozuk API key formatı ("AQ." öneki, bilinen
Google-tarafı sorunu) yaşandı. OpenRouter'da da varsayılan model
(`gpt-oss-120b`) canlı listeden düşmüştü. `check_openrouter_models.py`
scripti tam da bunun için yazıldı — tahmin etmek yerine anlık sorgula.

**Karar:** Claude API'ye ($5 minimum ödeme) hiç gerek kalmadı. OpenRouter'ın
ücretsiz Nemotron 3 Ultra modeli yeterli kaliteyi verdi.

### Adım 2b'de ikinci ölçüm: Nemotron vs Ollama (18 Ağu 2026)

Çoğaltmaya başlarken "hangi model" sorusu için ikisi **aynı görevde, aynı
prompt'la, aynı hedefle** (İstasyon Mekanik, 40 kayıt) ölçüldü:

| | Ollama qwen2.5:14b | OpenRouter Nemotron 3 Ultra |
| --- | --- | --- |
| gereken çağrı | 7 | **4** |
| süre | 6:45 | **2:11** |
| `review.py` işaretli | **%18** | **%5** |
| near-dup reddi | 5 | **0** |
| uydurma istasyon adı | `marmaraisi`, `yersenlik`, `leventtünel` | yok |
| uydurma teknik terim | `perde çarkı`, `motorı tıkranıyor` | yok |

**Nihai 1586 kayıt üzerinde de aynı fark doğrulandı:**

| kaynak | kayıt | işaretli |
| --- | --- | --- |
| openrouter (Nemotron) | 1086 | **%4.5** |
| ollama (qwen2.5:14b) | 500 | **%15.8** |

**Ders 2'nin ikinci teyidi.** Ayrıca yeni bir gözlem: qwen'in asıl zayıflığı
`review.py`'nin ölçtüğü şey (uzunluk/tekrar) değil, **özel isim ve teknik
terim uydurması** — otomatik triyaj bunu yakalamıyor, elle okumak gerekiyor.
Yani düşük işaretli oran tek başına kalite garantisi değil.

**`AMPLIFY_BATCH_SIZE` 25→40 kararı:** bağlayıcı kısıt örnek sayısı değil
ÇAĞRI sayısı. OpenRouter ücretsiz katmanı ~50 istek/gün; 25'lik partilerle
1600 örnek 64 çağrı gerektiriyordu, yani son ~350 kayıt zorunlu olarak
Ollama'ya kalıyordu. 40'lık partiyle ~40 çağrı yetiyor. (Pratikte 45
OpenRouter çağrısı yapılabildi, sonra kota bitti.)

**Hibrit devir sahada çalıştı:** kota `altyapi_insaat` ortasında bitti,
script otomatik Ollama'ya geçti ve durmadan devam etti. Ayrıca bir parti
bozuk JSON döndürdüğünde tüm çalıştırmayı çökertmek yerine o partiyi atlayıp
devam etti (kalıcı/geçici hata ayrımı — Genel İlke 5).

### Ollama verisinin Nemotron'la değiştirilmesi (19 Ağu 2026)

Kota bitince 500 kayıt (Yolcu/Operasyon ve Temizlik/Çevre'nin TAMAMI,
Altyapı/İnşaat'ın yarısı) qwen2.5:14b'den gelmişti. Sorun oranın kendisi değil
(%31), **dağılımın kategori bazında sistematik olması**: rastgele serpilse
zararsızdı, ama iki sınıf tamamen zayıf modelden gelince o sınıfların F1'i
gerçeği yansıtmaz ve confusion matrix yanıltır. Kota yenilenince değiştirildi.

`generate_data.py --replace-source ollama` ile yapıldı (aşağıda). Kategoriyi
tümden sıfırlamak yerine sadece hedef kaynağı silmek şart oldu: Altyapı/İnşaat
karışıktı (86 Nemotron + 100 Ollama), `--force` olsaydı 86 iyi kayıt da giderdi.

**`--provider openrouter` kullanıldı, hibrit DEĞİL.** Sebep: hibrit modda kota
yarıda bitse script sessizce Ollama'ya düşer ve tam da temizlenen veriyi geri
koyardı. Tek sağlayıcı denince kota bitiminde temiz şekilde durur.

Sonuç — 22 çağrı, işaretli oran kategori bazında:

| kategori | önce | sonra |
| --- | --- | --- |
| Altyapı / İnşaat | 186 kayıt, %6.5 | **200 kayıt, %0.5** |
| Yolcu / Operasyon | 200 kayıt, %7.5 | **200 kayıt, %6.0** |
| Temizlik / Çevre | 200 kayıt, %26.5 | **200 kayıt, %0.0** |
| **TOPLAM** | 1586 kayıt, **%8.1** | 1600 kayıt, **%3.8** |

Ollama'nın takıldığı `altyapi_insaat / devrik` grubu (36/50'de doyuma ulaşmıştı)
Nemotron tarafından sorunsuz tamamlandı — yani doygunluk modelin çeşitlilik
kapasitesiyle ilgiliydi, prompt'la değil.

**Cümle bazında kıyas** (eski sürüm `amplified_ollama_backup.jsonl`'de duruyor):

| | Ollama | Nemotron |
| --- | --- | --- |
| istasyon adı | `Beköy` (uydurma) | Topkapı, Esenler, Gayrettepe (gerçek) |
| anlam | `Döküntüler çıkışa`, `cevre zemin kaygan dogal yapi suyundan nedenli` | `Yağ lekesi kaygan vagon` |
| uydurma kelime | `anunci` (anons) | yok |
| ses/kip | `Sefer sayisini azalttik` (birinci şahıs, bildirim diline aykırı) | `Sefer seyreltildi` (edilgen) |
| detay | tekrarlı (`Döküntüler yarattı`, `Çöp birikmiş`) | somut (`asansör kabininde yazıcı tozu`) |

## review.py — Kalite Kontrol Sistemi

Otomatik olarak şunları işaretler (elle bakılması gerekenler):

- `DUP` — birebir tekrar
- `BENZER` — yakın kopya (SequenceMatcher + Jaccard kelime-kümesi hibrit,
  1.0'da sınırlı)
- `UZUNLUK` — stilin beklediği kelime aralığı dışında
- `SIZINTI` — kategori adını çağrıştıran kelime, kategori başına %25 payını
  aşıyor
- `SINIR` — başka KATEGORİDEKİ bir bildirime çok benziyor (etiket tutarsızlığı)
- `YABANCI` — Türkçe olmayan kelime şüphesi (`q/w/x` harfi veya Türkçe'de
  geçmeyen digraf). Bilgi amaçlı; dar ama kesin bir kural, kapsamlı değil.

**ASCII/aksan kontrolü bilgi amaçlıdır, işaretlenmez:** İlk sürümde "hiç
Türkçe karakter yok" tespit edilince işaretleniyordu, ama bu yanlış alarm
üretiyordu ("Turnike 3 bozuk" gibi zaten aksan gerektirmeyen doğru cümleler
de işaretleniyordu). Sonra `DIACRITIC_VOCAB` adında, `config.py`'nin kendi
doğru-yazılmış Türkçe metninden (kategori açıklamaları + istasyon adları)
otomatik çıkarılan bir sözlükle gerçek aksan-düşürme tespit edildi
(`Mecidiyekoy`→`Mecidiyeköy` gibi). Ama kullanıcı geri bildiriminden sonra
(gerçek hayatta klavye alışkanlığı olarak normal olduğu için) bu bulgular
"sorun" listesinden çıkarılıp sadece kategori özetinde "aksan-düşük%" olarak
bilgi amaçlı raporlanmaya çevrildi — **artık işaretli sayıya dahil değil.**

**`--file amplified` eklendi:** Adım 2b çıktısı da aynı araçla taranıyor.
Varsayılan taramaya dahil değil (1600 kayıtlık rapor seed/gold raporunu
boğar), açıkça seçilmesi gerekiyor.

**`similarity()` simetrik hâle getirildi (18 Ağu 2026, gerçek hata):**
`SequenceMatcher` argüman sırasına duyarlı (autojunk sezgiseli). Ölçüldü: bir
çift için `similarity(a,b) = 0.8511`, `similarity(b,a) = 0.8298` — yani 0.85
eşiğinin iki yanında. Sonuç: aynı çift, nerede karşılaştırıldığına göre farklı
karar alıyordu. `generate_data` `(yeni, mevcut)` sırasıyla çağırıp kabul
ederken, `preprocess` `(mevcut, yeni)` sırasıyla çağırıp aynı çifti
near-duplicate sayıyordu. Sızıntı savunmasının tamamı bu eşiğe dayandığı için
girdiler artık kanonik sıraya sokuluyor (`sorted((a, b))`); 2000 rastgele
çiftle simetri doğrulandı.

**`SINIR` bayrağı eklendi (19 Ağu 2026) — kategori sınırı artık denetleniyor:**
Önceki sürümde `review.py` bir bildirimin YANLIŞ kategoride olduğunu tespit
edemiyordu; sadece tekrar/uzunluk/sızıntı bakıyordu. Gold'daki yanlış
kategorili bir kayıt (peron kapısı PSD arızası `guvenlik_emniyet` etiketiyle)
ancak şans eseri "uzunluk" bayrağıyla yakalanmıştı — bu, aracın kör noktasıydı.

`SINIR`, **farklı kategorilerdeki** kayıtları birbiriyle karşılaştırır ve
`CLUSTER_THRESHOLD` (0.80) üstünde benzeyen çiftleri işaretler. Mantığı
`BENZER`'den ayrı: orada sorun tekrar, burada **etiket tutarsızlığı** —
neredeyse aynı metnin iki farklı etikette olması modele çelişkili sinyal verir
ve biri muhtemelen yanlış kategoridedir.

İlk çalıştırmada 1600 kayıtta **5 çift** buldu. Elle değerlendirildi:
- **3'ü gerçek etiket hatasıydı**, düzeltildi (aşağıda)
- **2'si yanlış alarm**: ölçüt sözcüksel olduğu için gerçekten farklı arızalar
  ortak kelimeler yüzünden yakalandı (`Asansör kabin titriyor` /
  `Asansör kabini kirli`; `Makinist masası acil durdurma butonu takılı` /
  `Acil durdurma butonu takılı`)

Düzeltme sonrası kalan: 2 çift (4 kayıt), ikisi de bilinen yanlış alarm.
Seed'de 1 yanlış alarm, **gold'da hiç yok**.

## generate_seed.py — `--category` desteği (18 Ağu 2026 eklendi)

Sorun: `resume` mantığı bir kategoriyi %80 eşiğini (8/10) geçtiğinde "tamam"
sayıyordu, bu yüzden 9/10 kalan `guvenlik_emniyet` otomatik tamamlanmıyordu
ve elle müdahale yolu da yoktu.

- `--category KEY [KEY ...]` — sadece o kategori(ler)i işler, **%80 eşiği
  yerine tam hedefe** tamamlar, diğer kategorilere hiç dokunmaz.
- `--force` artık `--category` ile birlikte **yalnızca seçili kategoriyi**
  sıfırlar (tek başına kullanıldığında eski davranış: hepsini sıfırlar).
- Tamamlamada **az temsil edilen stiller önceliklendirilir** — böylece
  tamamlama stil dengesini bozmak yerine düzeltir.
- Rapor artık sessiz kalmıyor: eşiği geçmiş ama hedefin altındaki kategoriler
  `<-- 1 eksik (--category X ile tamamlanabilir)` diye işaretleniyor. Asıl
  sorun bu boşluktu.

## Elle Yapılan Son Düzeltmeler (apply_review.py ile)

v4 (Nemotron) verisi üzerinde elle triyaj yapıldı, onaylanan değişiklikler:

**Seed:**

- 3 kayıt silindi: anlamsız "fren manası" ifadesi, bir yakın-kopya, bir
  birebir kopya
- 1 kayıt bilerek **silinmedi**: "Turnike 5 mekanik arıza" — kategori adını
  çağrıştırsa da gerçekçi bir kısa bildirim olduğu değerlendirildi
- 1 stil düzeltmesi: aşırı resmi/uzun bir "devrik" örneği → `standart`
  olarak yeniden etiketlendi (metne dokunulmadı)
- 1 anlam düzeltmesi: "...seferler **gecikmemiştir**" → "...seferler
  **gecikmiştir**" (anlam ters dönmüştü)

**Gold:**

- 4 stil düzeltmesi: `elektrik_enerji` kategorisinde `devrik`/`cok_kisa`
  etiketli ama aslında tam resmi cümle olan 4 kayıt → `standart`

### İkinci tur (18 Ağu 2026)

**Seed:**

- `makinist kabini fren manasi tikaniyo` → `...fren manivelasi tikaniyo`.
  "Fren manası" diye bir parça yok; aynı ifadenin `standart` stildeki ikizi
  ilk turda silinmişti ama bu `yazim_yanlisi` varyantı gözden kaçmıştı.
  Few-shot yemi olduğu için hatalı terimi 1600 örneğe taşıma riski vardı.

**Gold:**

- `guvenlik_emniyet` kategorisi `--force` ile tamamen yeniden üretildi.
  Sebep: 9 kaydın 3'ü bozuktu — biri **Arapça harf** içeriyordu
  (`Peron kapısı arıza, güvenlik بوğu`), biri var olmayan bir kelime
  kullanıyordu (`yangın merchı`), biri çatı hatası taşıyordu
  (`güvenlik ekipleri bölgeyi kuşatıldı`). Yedek:
  `gold_v4_pre_guvenlik_backup.jsonl`
- Yeniden üretim sonrası 1 kayıt silindi: `Peron kapısı aralıklı açılıyor,
  sıkışma riski var.` — taksonomiye göre **yanlış kategori** (config'de
  "peron kapısı (PSD)" açıkça `istasyon_mekanik` kapsamında). Gold'da etiket
  hatası doğru tahmini yanlış saydırır, bu yüzden yazım hatasının aksine
  tolere edilmez. Yerine 1 kayıt üretildi.
- 2 stil düzeltmesi (metne dokunulmadan).

**Bilinçli olarak DÜZELTİLMEYENLER (kullanıcı kararı):** `trensformatörü`
(transformatörü), `kırıkMetal` (yapışık kelime), `Ayrılıkçeşmesi` (istasyon
adı kısaltması). Ölçüt: *"Metro İstanbul'da bir personel aceleyle yazarken
bunu yazar mıydı?"* — evetse gerçekçi gürültüdür, gold'a aittir. Ayrıca
teknik olarak metriğe sadece `metin` + `kategori` giriyor; `stil` etiketi
muhasebe alanı, model `kategori` tahmin ediyor.

### Üçüncü tur — seed'in elle okunması (19 Ağu 2026)

93 kaydın tamamı tek tek okundu. Bulunan 3 **var olmayan kelime** düzeltildi:

| eski | yeni | neden |
| --- | --- | --- |
| `giriş tornası` | `giriş turnikesi` | "torna" tezgâh demek |
| `merkeziyete sinyal` | `merkeze sinyal` | "merkeziyet" böyle kullanılmaz |
| `betonarme kırışması` | `betonarme kırılması` | "kırışmak" buruşmak demek |

**Karar ölçütü — gerçekçi yazım hatası ile stil sözleşmesi ihlali farklı
şeylerdir.** Üçü de `standart`/`devrik` etiketliydi; config bu stiller için
doğru Türkçe yazımı ZORUNLU tutuyor (`generate_seed.py` prompt kural 9:
"SADECE `yazim_yanlisi` stilinde harf düşür, diğer üç stilde doğru yazım
zorunludur"). "İnsan da böyle yazabilir" doğru bir itiraz, ama o insan
`yazim_yanlisi` stilinde yazıyor demektir. Yani ölçüt "hata gerçekçi mi"
değil, **"kaydın kendi stil etiketi hataya izin veriyor mu"**.

**Bilinçli olarak DEĞİŞTİRİLMEYEN kategori (kullanıcı kararı):** seed #75
`Kart okumuyor ucret odendi bilet alamadi.` → `yolcu_operasyon` kaldı.
Config'in turnike kuralı "kart okumama → `yazilim_sistem`" diyor, ama
kullanıcının gerekçesi: bildirimin öznesi okuyucu arızası değil, **ücret
ödemiş yolcunun mağduriyeti**. Not: bu yorum config metniyle gerginlik
taşıyor; ileride yeniden üretim yapılırsa LLM config'i takip edeceği için
bu kayıt tek başına kalabilir.

Not: çoğaltma zaten tamamlandığı için bu düzeltmeler mevcut 1600 kaydı
DEĞİŞTİRMEZ. Değeri ileride yeniden üretim yapılırsa veya `--include-seed`
ile seed eğitime katılırsa ortaya çıkar.

## ✅ Kapanan Açık Noktalar (18 Ağu 2026)

Önceki sürümdeki 5 maddenin tamamı kapandı:

1. **`apply_review.py` doğrulandı** — daha önce çalıştırılmıştı; 9
   değişikliğin her biri dosyada tek tek kontrol edilerek teyit edildi.
   Ayrıca script tekrar çalıştırılabilir (idempotent) hâle getirildi:
   "zaten uygulanmış" ile "hiç bulunamadı" ayırt ediliyor, sahte UYARI
   basmıyor.
2. **Gold'daki eksik kategori tamamlandı** — `generate_seed.py`'ye
   `--category KEY` eklendi (aşağıda). `guvenlik_emniyet` `--force` ile
   yeniden üretildi, gold 80/80 oldu.
3. **`SEED_PROVIDER = "openrouter"` teyit edildi** (config.py).
4. **qwen2.5:14b vs gemma2:9b karşılaştırması hâlâ yapılmadı** ve artık
   gereksiz: Adım 2b'de qwen2.5:14b ile Nemotron doğrudan ölçüldü, qwen
   belirgin şekilde geride kaldı. Daha küçük bir yerel modeli denemenin
   kazandıracağı bir şey yok.
5. **Adım 2b model kararı verildi:** hibrit (aşağıda).

## ⚠️ Güncel Açık Noktalar

1. ✅ **KAPANDI — `altyapi_insaat` 200/200.** Nemotron `devrik` grubunu
   tamamladı (Ollama 36/50'de doyuma ulaşmıştı).
2. ✅ **KAPANDI — Ollama'dan gelen 500 kayıt değiştirildi** (19 Ağu, yukarıda
   detaylı). Veri artık %100 Nemotron. Yedek `amplified_ollama_backup.jsonl`
   olarak duruyor, `preprocess.py` yeniden çalıştırıldı.
3. ✅ **KAPANDI — near-dup eşiği kalibre edildi (19 Ağu).** İki ayrı eşiğe
   ayrıldı: `NEAR_DUP_THRESHOLD = 0.85` (üretim) ve `CLUSTER_THRESHOLD = 0.80`
   (bölme). Gerekçe ve ölçüm `config.py`'de detaylı.
4. ✅ **KAPANDI — kategori sınırı kontrolü eklendi (`SINIR` bayrağı).**
   Bulduğu 3 gerçek etiket hatası düzeltildi.
5. ✅ **KAPANDI — taksonomi belirsizliği confusion matrix'te incelendi.**
   İşaretlenen `arac_tren ↔ guvenlik_emniyet` çifti **hiç karışmadı**; model
   ayırt etti. Yerine gerçek bir çakışma bulundu (`guvenlik_emniyet ↔
   yolcu_operasyon`, config kaynaklı) ve buna kural yerine **ikincil kategori
   mekanizması** getirildi (Adım 4 bölümünde detaylı).
6. ✅ **KAPANDI — `CONFIDENCE_THRESHOLD` kalibre edildi**, 0.60 → **0.70**.
7. **Kategori dengesi artık tam eşit değil:** `SINIR` düzeltmeleri sonrası
   istasyon_mekanik 202, altyapi_insaat ve yolcu_operasyon 199, diğerleri 200.
   Sapma ±2, macro-F1 ve katmanlı bölme için önemsiz — ama raporda "her
   kategoriden tam 200" denmemeli.
8. ✅ **KAPANDI — aksan dayanıklılığı çözüldü** (19 Ağu). ASCII çoğaltmayla
   aksan kaybı −6.36 puandan **−1.16 puana** indi; genel doğruluk da arttı.
   Detay Adım 4 bölümünde.
9. ✅ **KAPANDI — `YABANCI` bayrağı eklendi + 9 kayıt düzeltildi.** Kuralın
   kapsamlı olmadığı (dar ama kesin) açıkça belgelendi.
10. **İkincil kategori mekanizması Adım 5/6'da hayata geçirilecek** —
    `MARGIN_THRESHOLD` config'te hazır, backend `/predict` ve arayüz bunu
    kullanacak.
11. **`--include-seed` hâlâ kapalı.** 93 kayıtlık seed eğitime katılmıyor.
    Katılırsa küçük bir kazanç olabilir; ölçülmedi. Seed'in few-shot yemi
    olması eğitime girmesine engel değil (gold farklı, o asla girmez).

## Yol Haritası — Kalan Adımlar

**✅ Adım 2b — Çoğaltma (TAMAMLANDI):** `src/generate_data.py` yazıldı ve
çalıştırıldı, 1600 kayıt üretildi (19 Ağu değişimi sonrası %100 Nemotron).

- Few-shot **yalnızca** `seed.jsonl`'den; `gold.jsonl` bu dosyada hiç okunmuyor.
- Her çağrı **tek (kategori, stil)** ikilisi için. Sebep: Adım 2a'da en sık
  hata uzunluk kuralına uymamaktı; tek stil isteyince model aynı anda dört
  farklı uzunluk aralığını yönetmek zorunda kalmıyor.
- Few-shot iki başlığa ayrıldı: *stil örnekleri* (uzunluğu öğretir) ve *konu
  örnekleri* (kategoriyi öğretir, "uzunluğunu taklit etme" notuyla). Karışık
  gösterildiğinde model yanlış uzunluk sinyali alıyordu.
- Çeşitlilik üç katmanda zorlanıyor: her çağrıda rastgele `SLOT_VALUES`
  enjeksiyonu, üretilmişlerden örneklem ile "bunları tekrarlama" listesi, ve
  eklemeden önce `review.similarity` ile near-dup reddi.
- CLI: `--provider {hybrid,openrouter,ollama}`, `--category`, `--target`,
  `--dry-run` (LLM çağırmadan prompt'u yazdırır — kota harcamadan test için),
  `--replace-source {ollama,openrouter}`.
- **`--replace-source`** belirtilen sağlayıcıdan gelen mevcut kayıtları silip
  yerine yenisini ürettirir. Kategoriyi tümden sıfırlamaya göre avantajı:
  karışık kategorilerde iyi kayıtlar korunur. Silinen kayıtlar near-dup
  havuzundan da çıkar, böylece yeni model aynı konuları serbestçe yazabilir.
  `--dry-run` ile birlikte kullanılınca ne silineceğini dosyaya dokunmadan
  gösterir.

**✅ Adım 3 — Ön İşleme (TAMAMLANDI):** `src/preprocess.py` yazıldı ve
çalıştırıldı (5.6 sn).

- Near-duplicate **kümeleme split'ten önce** yapılıyor (union-find,
  `CLUSTER_THRESHOLD`=0.80), bir kümenin tüm üyeleri hep aynı bölmede kalıyor — çoğaltılmış verinin
  train/test'e sızıp sahte yüksek doğruluk üretmesini engelleyen asıl
  mekanizma. `review.similarity` ile AYNI ölçüt kullanılıyor.
- Katmanlı bölme: her kategori kendi içinde bölünüyor, sınıf dengesi üç
  bölmede de korunuyor. Sonuç tam %80/%10/%10.
- Temizlik: boş/uzunluk dışı kayıtlar, birebir tekrarlar ve **etiket
  çakışmaları** (aynı metin iki farklı kategoride → ikisi de atılır) eleniyor.
- **Gold sızıntı kontrolü her çalıştırmada otomatik**: hiçbir gold metni
  eğitim havuzunda olmamalı; doğrulandı, temiz.
- Raporda kaynak dağılımı da var (hangi modelin verisi hangi bölmeye düştü).
- CLI: `--include-seed` (seed.jsonl'i de eğitime katar; şu an KAPALI),
  `--report-only` (dosya yazmadan rapor).
- Doğrulandı: label↔kategori uyumsuzluğu 0, bölmeler arası birebir kesişim 0,
  gold ↔ train/test kesişim 0.

### Near-dup eşiği kalibrasyonu (19 Ağu 2026)

**Kalibrasyonun tuzağı:** veri zaten 0.85 eşiğiyle filtrelenmiş üretildi, yani
0.85 üstü çift tanım gereği yok. Mevcut veriye bakarak "eşik doğru mu"
sorusu cevaplanamaz — **eşiğin ALTINDAKİ banda** bakmak gerekiyor. Orada
gerçek kopyalar varsa eşik fazla gevşek demektir.

Aynı kategori içindeki 159.200 çiftin dağılımı:

| bant | çift |
| --- | --- |
| 0.60-0.70 | 1487 |
| 0.70-0.75 | 191 |
| 0.75-0.80 | 118 |
| 0.80-0.85 | 25 |
| 0.85+ | 1 (simetri düzeltmesinden önce kaçan tek çift) |

0.80-0.85 bandı elle okundu ve **gerçek anlamsal kopyalar bulundu**:
`Taksim istasyonunda 4 numaralı vagondaki yolcu anons cihazı ses vermiyor.` /
`Yolcu anons cihazı ses vermiyor 4. vagon` (0.843) — aynı arıza, aynı vagon.
Ama aynı bantta **gerçekten farklı arızalar** da var: `makinist kabini sağ
tarafı ayna kırık` / `makinist kabini saati durmuş` (0.804). Ölçüt sözcüksel,
anlamsal değil; eşiği körü körüne düşürmek farklı arızaları birleştirir.

**Çözüm: tek eşik yerine iki eşik**, çünkü eşiğin iki farklı işi var ve hata
maliyetleri simetrik değil:

| kullanım | yanlış birleştirme | kaçırma |
| --- | --- | --- |
| üretim (yeni kayıt reddi) | iyi cümle boşa gider, kota harcanır | veri biraz tekrarlı olur |
| bölme (kümeleme) | iki kayıt aynı bölmeye düşer, küçük çeşitlilik kaybı | **metrik şişer, sahte başarı** |

Bölmede kaçırmanın bedeli çok daha ağır → orada daha agresif olunmalı.

Kümeleme etkisi: 0.85→1 çift, 0.82→8, **0.80→27 çift + 1 üçlü**, 0.78→39
(kümeler 5'e zincirlenmeye başlıyor), 0.75→88 (fazla agresif). **0.80
seçildi:** küme boyutu 2-3'te kalıyor, bedeli 1600 kayıtta 29 kayıt.
Yakaladığı en büyük küme tam da gerçek kopya ailesi (üç ayrı "sefer iptali →
yolcular bir sonraki trene yönlendirildi" cümlesi).

**✅ Adım 4 — Model Eğitimi (TAMAMLANDI):** `src/train.py` + `src/evaluate.py`
yazıldı ve çalıştırıldı. **Tüm başarı kriterleri her iki test setinde de
geçildi.**

| metrik | test (160) | gold (80) | hedef |
| --- | --- | --- | --- |
| accuracy | 0.9125 | **0.9250** | 0.85 ✅ |
| macro F1 | 0.9135 | **0.9247** | 0.82 ✅ |
| en düşük sınıf F1 | 0.8293 | **0.8000** | 0.75 ✅ |

(Bu değerler ASCII çoğaltmalı ikinci eğitimden. Çoğaltmasız ilk eğitim:
test 0.8938/0.8935/0.7500, gold 0.9000/0.9014/0.8000 — o da tüm hedefleri
geçiyordu, çoğaltma hepsini yukarı taşıdı.)

**Gold skoru test'ten YÜKSEK (+0.011).** Projenin en önemli bulgusu bu:
sentetik veriyle eğitilen model, bağımsız üretilmiş ve elle gözden geçirilmiş
gold setinde en az kendi dağılımı kadar iyi. Yani model çoğaltmanın kalıplarını
ezberlememiş, gerçekten sınıfı öğrenmiş. "Sentetik veri gerçekçi mi"
eleştirisine verilebilecek en somut cevap.

### En kritik hata: LoRA'da öğrenme hızı

`config.py`'de `LEARNING_RATE = 2e-5` yazıyordu ve **model hiçbir şey
öğrenmiyordu**: 5 epoch sonunda val macro-F1 = 0.134, kayıp 2.073 (rastgele
seviye `ln(8) = 2.079`). Sebep: 2e-5 BERT'i **tam fine-tuning** ederken
kullanılan standart değer, ama biz LoRA kullanıyoruz — parametrelerin sadece
%0.54'ü (595.976 / 111.219.472) eğitiliyor, adaptörler sıfırdan başlıyor ve
sınıflandırma başlığı rastgele başlatılıyor. Bu kadar küçük bir öğrenme hızıyla
ağırlıklar anlamlı mesafe kat edemiyor.

| LR | val macro-F1 (5 epoch) |
| --- | --- |
| 2e-5 | 0.134 (rastgele) |
| 1e-4 | 0.393 |
| 3e-4 | 0.850 |
| 5e-4 | 0.875 |

15 epoch'ta: 5e-4 → 0.930, 1e-3 → 0.938. Aradaki fark val setinde ~1 örnek
(n=160), yani gürültü içinde — 5e-4 seçildi (daha yumuşak eğri).

**Ders: hiperparametreyi literatürden kopyalamak yetmiyor, eğitim yöntemine
göre ayarlamak gerekiyor.** Rapor için güçlü bir bölüm.

### Diğer eğitim kararları

- **HuggingFace `Trainer` yerine elle eğitim döngüsü.** `Trainer`, `accelerate`
  üzerinden MPS'te dtype/device sürprizleri çıkarabiliyor ve hata ayıklamayı
  zorlaştırıyor. 1280 örnek × 12 epoch için elle döngü hem şeffaf hem yeterli.
- **`modules_to_save=["classifier"]`** — LoRA'da kritik: sınıflandırma başlığı
  rastgele başlatılıyor, sadece adaptörler eğitilirse model öğrenemez.
- **En iyi val macro-F1 veren epoch kaydediliyor**, sonuncusu değil (son epoch
  genelde aşırı öğrenmiş olur). Seçilen: epoch 5, val F1 0.9242.
- Tohum sabit (`SEED=42`), sonuç tekrar üretilebilir.
- Eğitim süresi ~35-45 sn/epoch (MPS, Apple Silicon; çoğaltmayla train 2219
  kayda çıktı). LoRA çıktısı sadece **2.4 MB** — tam model 440 MB olurdu.
- Seçilen: epoch 6, val macro-F1 **0.9429**.

### Confusion matrix bulguları

**Beklenen karışma GERÇEKLEŞMEDİ.** Adım 3'te işaretlenen taksonomi
belirsizliği (`Makinist masası acil durdurma butonu takılı` train'de arac_tren,
`Acil durdurma butonu takılı` test'te guvenlik_emniyet, benzerlik 1.00) için
"model puan kaybedecek" denmişti. Model **güven 1.00 ile doğru** cevap verdi;
"makinist masası" ifadesinin ayırt edici sinyal olduğunu öğrenmiş. Confusion
matrix'te `arac_tren ↔ guvenlik_emniyet` karışması hiç yok.

**Bunun yerine gerçek bir çakışma ortaya çıktı: `guvenlik_emniyet ↔
yolcu_operasyon`** (test'te 5, gold'da 2 hata — her iki sette de baskın çift).
Kaynağı veri değil, **config'in kendisi**:
- `guvenlik_emniyet` kapsamı: "...anons ile tahliye"
- `yolcu_operasyon` kapsamı: "anons yapılmaması/yanlış anons"

Örnek: `Acil tahliye anonsu yoğun saatlerde peronda net duyulamıyor.` — iki
kapsama da giriyor. Çözüm için aşağıdaki ikincil kategori mekanizması seçildi.

### Aksan dayanıklılığı — ölçüm, teşhis, çözüm, doğrulama

İlk eğitimden sonra `yazim_yanlisi` stilinin zayıf göründüğü fark edildi
(gold 0.765). Ama bu tek başına yanıltıcıydı: test+gold birleşik ölçümde
0.852 ve %95 güven aralığı diğer stillerle **fazlasıyla örtüşüyordu**
([0.763-0.941] vs [0.858-0.988]) — n=61 ile istatistiksel anlamlılık yok.

Bu yüzden **nedensel test** yapıldı: test+gold'daki aksan içeren 173 kaydın
aksanları kaldırılıp yeniden tahmin edildi. İçerik aynı, sadece ç/ğ/ı/ö/ş/ü
düşürüldü:

| | doğruluk |
| --- | --- |
| orijinal (aksanlı) | 157/173 = **0.9075** |
| ASCII katlanmış | 146/173 = **0.8439** |

**6.4 puanlık kayıp, tek değişken aksan.** Stil etiketine bakmak gürültülüydü,
müdahaleli test net cevap verdi.

**Teşhis — BERTurk tokenizer'ında görünüyor:**
```
asansör  -> 1 parça  ['asansör']
asansor  -> 3 parça  ['asa', '##ns', '##or']
```
Aksan düşünce kelime anlamsız alt-parçalara bölünüyor.

**Çözüm: eğitim verisi çoğaltma.** `train.csv`'deki aksan içeren kayıtların
ASCII'ye katlanmış kopyaları eğitime eklendi (1280 → **2219**, +939 kopya).
API gerektirmez. Sızıntı riski yok: `preprocess`'teki kümeleme zaten
aksan-duyarsız (`review.normalize` aksanı kaldırıyor), yani bir train kaydının
ASCII kopyası test'teki bir kayıtla eşleşiyorsa o ikisi zaten aynı kümededir.

**Doğrulama — aynı nedensel test tekrarlandı:**

| | çoğaltmasız | çoğaltmalı |
| --- | --- | --- |
| orijinal (aksanlı) | 0.9075 | **0.9191** |
| ASCII katlanmış | 0.8439 | **0.9075** |
| **aksan kaybı** | **−6.36 puan** | **−1.16 puan** |

Dayanıklılık 5.2 puan iyileşti. Ayrıca **genel doğruluk da arttı** (test macro
F1 0.8935 → 0.9135, gold 0.9014 → 0.9247) — çoğaltma sadece aksan sorununu
çözmedi, model genelinde fayda sağladı. `config.AUGMENT_ASCII_FOLD` ile
kapatılabilir (`--no-augment`), kıyas yapmak için.

### Yabancı kelime tespiti — üç yöntem denendi, ikisi başarısız

Eğitim verisinde `Acil çıkış yolu engelli baggage` bulundu (İngilizce kelime).
Otomatik tespit için denenenler:

| yaklaşım | sonuç |
| --- | --- |
| Projenin kendi metninden bigram sözlüğü | ❌ 1600 kayıtta **355 yanlış alarm** — referans korpus (247 cümle) Türkçe'nin bigram uzayını kapsayamıyor; `nesne`, `açma`, `sessiz` işaretlendi |
| Geniş digraf listesi (`sh`, `th`, `ph`, `ay`...) | ❌ `şüpheli`, `Kağıthane`, `aydınlatma` gibi Türkçe kelimeleri yakalıyor |
| BERTurk tokenizer parça sayısı | ❌ ayırt edemiyor: `baggage` 3 parça, `asansor` da 3 parça |
| **Dar kural: `q/w/x` + `ck,gh,ea,oo`** | ✅ 8 gerçek bulgu, **0 yanlış alarm** — ama kapsamlı değil, `baggage`'ı kaçırıyor |

Türk alfabesinde q, w, x **yok** — bu kısım kesin. Dar kural `YABANCI` bayrağı
olarak `review.py`'ye eklendi ama **bilgi amaçlı**: `switch`, `wifi` gibi
kelimeler Türkçe teknik jargonda da kullanılıyor, bayrak "sil" değil "bak"
demek.

Bulunan 9 kayıt (8 otomatik + `baggage` elle) düzeltildi: `bearing`→rulmanı,
`wiper`→silecek, `switch`→anahtarı, `watchdog`→izleme servisi, `WiFi`→kablosuz
ağ, `baggage`→bagajla, `duraqta`→durakta, `kapaq`→kapak (×2).

**Ders:** otomatik tespit her sorun için mümkün değil. Türkçe sözlük olmadan
"bu kelime Türkçe mi" sorusu güvenilir cevaplanamıyor; elle okuma hâlâ tek
kapsamlı yöntem. Dar ve kesin bir kural, geniş ve gürültülü bir kuraldan iyidir.

### İkincil kategori mekanizması — sınır sorunlarına genel çözüm

Taksonomiye sınır kuralı yazmak yerine (8 kategoride 28 çift var, ölçeklenmez)
modelin **zaten ürettiği** bilgi kullanılıyor:

| | test | gold |
| --- | --- | --- |
| top-1 doğruluk | 0.913 | 0.925 |
| **top-2 doğruluk** | **0.963** | **0.975** |
| marj eşiği 0.40'ta çift kategorili dönen | %6.2 | %7.5 |

Model belirsiz olduğunu biliyor. `MARGIN_THRESHOLD = 0.40` altında `/predict`
birincil + ikincil kategori döner. Kalibrasyon (gold): 0.30 → 3 hata kurtarılır
1 boşuna; **0.40 → 4 kurtarılır 1 boşuna**; 0.50 → 4 kurtarılır 3 boşuna.
0.40'tan sonra kurtarma artmıyor, maliyet artıyor.

Bunun kural yazmaya üstünlüğü: bugün bilmediğimiz sınır sorunlarını da kapsıyor,
gerçeği daha doğru modelliyor (bazı bildirimler gerçekten iki kategoriye girer),
ve mevcut `low_confidence` mekanizmasını kapsıyor.

**Adım 5 — Backend:** `backend/main.py`, FastAPI. `/predict` endpoint.
Model Python tarafında yüklü tutulacak, REST API olarak servis edilecek.
Dönecek alanlar:
- `kategori` + `guven` (birincil tahmin)
- `low_confidence: true` — güven `CONFIDENCE_THRESHOLD` (0.70) altındaysa
- `ikincil_kategori` — marj `MARGIN_THRESHOLD` (0.40) altındaysa; sınırda
  bildirimlerde ikinci ekip de bilgilendirilsin diye
- tüm kategorilerin olasılık dağılımı (arayüzdeki bar chart için)
- yanıt süresi

Model yüklemesi PEFT gerektiriyor: `adapter_config.json`'daki taban modelden
BERTurk indirilip üstüne LoRA adaptörü bindiriliyor (bkz. `evaluate.model_yukle`,
aynı desen kullanılabilir).

**Adım 6 — Frontend:** `frontend/`, React. Metin giriş, "Analiz Et" butonu,
kategori çıktısı (renk etiketiyle — `CATEGORY_COLOR` config'te hazır), güven
skoru progress bar, tüm kategorilerin olasılık dağılımı yatay bar chart,
örnek cümle listesi (tek tıkla doldurma), yanıt süresi göstergesi, düşük
güven uyarısı (`LOW_CONFIDENCE_MESSAGE` config'te hazır).

## Genel İlkeler (her adımda geçerli)

1. **`config.py` tek doğruluk kaynağı.** Yeni bir modül yazarken kategori/
   stil/hiperparametre asla orada yeniden tanımlanmaz, hep import edilir.
2. **Kaynak koddaki Türkçe metin her zaman doğru aksanlarla yazılır**
   (yukarıdaki "Ders 1"). Bu, LLM prompt'u olarak kullanılacak her yeni
   metin için geçerli (örn. Adım 2b'nin çoğaltma prompt'u).
3. **Gold seti few-shot'ta veya eğitimde asla kullanılmaz**, sadece nihai
   test için saklanır.
4. **Yeni bir LLM sağlayıcı denerken önce canlı model listesini sorgula**
   (`check_openrouter_models.py` örnek alınabilir), id tahmin etme.
5. **Kalıcı hata (kota/kimlik doğrulama) ile geçici hata (bozuk JSON, tek
   seferlik) ayrımı önemli** — `generate_seed.py`'deki `call_llm` bu ayrımı
   yapıyor, yeni script'lerde de aynı desen izlenmeli (tek kategori/parça
   hatası tüm çalıştırmayı çökertmemeli).
6. **`.gitignore`'a bir şey eklemeden ÖNCE gerçek boyutuna ve rolüne bak.**
   Bu projede aynı hata iki kez yapıldı: önce `data/raw/` + `data/processed/`,
   sonra `model/*.safetensors` refleks olarak gitignore'a eklendi ve ikisi de
   geri alınmak zorunda kaldı. İkisi de **adımların asıl çıktısıydı**, üstelik
   küçüktü (veri 580 KB, LoRA adaptörü 2.3 MB).

   Ölçüt — gitignore'a sadece şunlar girer:
   - yeniden üretilebilen ara ürünler (`__pycache__`, `*.pyc`)
   - gizli bilgi (`.env`)
   - makineye özel yerel ayarlar (`.claude/settings.local.json`, `.DS_Store`)
   - **gerçekten** büyük dosyalar (yüzlerce MB+)

   Bir adımın teslim ettiği ürün (üretilen veri, eğitilmiş model, bölünmüş
   veri seti) küçükse **commit edilir**. Sebep: bu proje adım adım ilerliyor ve
   her commit'in o adımın çalıştığını kanıtlaması gerekiyor; depoyu klonlayan
   birinin modeli yeniden eğitmek zorunda kalmaması lazım. Ayrıca LoRA
   adaptörünün 2.3 MB olması raporda öne çıkarılan bir bulgu — onu gizlemek
   kendi iddiamızın kanıtını silmek olurdu.

   **Dosya uzantısına bakıp varsayma, `du -h` ile bak.**

## Git İş Akışı Kuralı (Claude Code bunu her adımda uygulasın)

Bu proje adım adım (Adım 2b, 3, 4, 5, 6...) ilerliyor. Her adım için şu sıra
**kesinlikle** izlenir:

1. **Kodu yaz/düzenle.**
2. **Çalıştır ve test et.** Sadece kod yazmak yeterli değildir — script'i
   gerçekten çalıştır, çıktısını gör, hata varsa düzelt, tekrar çalıştır.
   "Kodu düzenledim" ile "denedim ve çalıştığını doğruladım" farklı
   şeylerdir; sadece ikincisi bir adımı tamamlanmış sayar.
3. **Bu dosyayı (`CLAUDE.md`) güncelle.** Bu ayrı bir iş veya sonraya
   bırakılabilir bir ek DEĞİL — adımın tamamlanma tanımının parçası.
   Güncellenecek tipik yerler: klasör yapısı (yeni yazılan dosya artık "YOK"
   görünmesin), veri sayıları, yeni config ayarları ve CLI argümanları,
   alınan kararlar ve gerekçeleri, ölçüm sonuçları, ve artık geçerli olmayan
   "Açık Noktalar" maddelerinin kapatılması. Sebep: bu dosya sıradan bir
   README değil, oturumlar arası taşınan tek bağlam kaynağı ve staj
   sunumunun dayanağı; güncellenmezse bir sonraki oturum yanlış bilgiyle
   başlar.
4. **Adım gerçekten çalıştığı doğrulanınca**, bana kısaca ne yapıldığını ve
   test sonucunu özetle, **git'e commit + push için onay iste.** Kod
   değişikliği ile `CLAUDE.md` güncellemesi AYNI commit'te sunulur.
5. **Benden açık onay gelmeden asla `git push` yapma.** "Onaylıyorum",
   "push'la", "evet" gibi net bir cevap bekle. Onay gelmeden bir sonraki
   adıma da geçme — sırayla ilerle.
6. Onay gelince o adımı **kendi başına bir commit** olarak işle (önceki
   adımlarla birleştirme) ve push et. Commit mesajı hangi adım olduğunu
   ve ne yapıldığını açıkça belirtsin (ör. `Adım 2b: Ollama ile çoğaltma
   scripti (generate_data.py) + 1600 örnek üretimi`).
7. Böylece git geçmişinde proje adım adım, her biri çalıştığı doğrulanmış
   halde görünür — bu hem staj sunumunda ilerlemeyi göstermek hem de bir
   adımda sorun çıkarsa geri dönebilmek için önemli.

**İlk push'tan önce:** Eğer bu klasörde henüz git deposu yoksa veya uzak
(remote) repo bağlı değilse, sessizce varsaymadan önce bana sor — hangi
remote'a (GitHub vb.) push edeceğimizi netleştirelim.

**Asla yapılmayacaklar:**

- Test edilmemiş kodu commit/push etmek
- Onay istemeden push etmek
- Birden fazla adımı tek commit'te birleştirmek
- Onay bekliyorken sessizce bir sonraki adıma geçmek

## Staj Sunumu İçin Notlar

- Orijinal PDF taslağındaki 6 kategori / 70 örnek/kategori / 420 toplam
  rakamları güncel: **8 kategori / 200 örnek/kategori / 1600 toplam.**
- PDF'te olmayıp sonradan eklenen: confidence threshold mekanizması
  (`low_confidence` uyarısı).
- 4 farklı LLM sağlayıcısının ampirik karşılaştırılması (yukarıdaki tablo)
  metodoloji bölümü için güçlü, özgün bir içerik — "neden bu modeli
  seçtim" sorusuna veriye dayalı bir cevap.
- Sentetik veri + gold test seti ayrımı, "veri gerçekçi mi" eleştirisine
  karşı somut bir savunma sağlıyor.
- **İkinci ampirik karşılaştırma (Adım 2b):** aynı görevde Nemotron %4.5 vs
  qwen2.5:14b %15.8 işaretli, 1586 kayıt üzerinden. Bulut/yerel model
  tercihini veriye dayandıran ikinci bir tablo.
- **"Düşük işaretli oran ≠ kalite" bulgusu:** qwen'in asıl zayıflığı otomatik
  triyajın ölçtüğü şey değil, özel isim/teknik terim uydurmasıydı
  (`marmaraisi`, `perde çarkı`). Otomatik metriklerin kör noktası olduğunu
  gösteren somut bir örnek — rapora olgunluk katar.
- **Veri sızıntısına karşı üç katmanlı savunma** anlatılabilir: (1) üretim
  anında near-dup reddi, (2) split öncesi kümeleme, (3) her çalıştırmada
  otomatik gold sızıntı kontrolü. Ayrıca kümeleme eşiğinde bulunup düzeltilen
  simetri hatası, "eşiğe dayanan sistemde ölçütün tutarlılığı kritiktir"
  dersinin somut örneği.
- **Gold'da yazım hatası bilinçli olarak korundu** ("personel aceleyle bunu
  yazar mıydı?" ölçütü). Gerçekçi gürültü kalır, üretim artığı temizlenir.
  Gold'u gerçek hayattan temiz yapmak başarı oranını şişirirdi.
- **En güçlü tek sonuç: gold skoru test'ten yüksek çıktı** (macro F1 **0.9247**
  vs 0.9135). Sentetik veriyle eğitilen model, bağımsız üretilmiş ve elle gözden
  geçirilmiş sette daha iyi. "Model ezberledi mi?" sorusuna ölçülmüş cevap.
  Üstelik bu iki bağımsız eğitimde de tekrarlandı (çoğaltmasız: 0.9014 vs
  0.8935), yani tesadüf değil.
- **LoRA öğrenme hızı hatası** (2e-5 → 5e-4, model rastgele seviyeden 0.93'e)
  metodoloji bölümü için değerli: hiperparametre literatürden kopyalanamaz,
  eğitim yöntemine göre ayarlanır. Ölçüm tablosu elde var.
- **Taksonomi sınır sorununa mühendislik çözümü:** kural yazmak yerine modelin
  olasılık dağılımını kullanmak (top-2 doğruluk 0.975). Kural bazlı çözümün
  neden ölçeklenmediği (28 kategori çifti) ve marj eşiğinin nasıl kalibre
  edildiği anlatılabilir. "Sistemi modele uydurmak yerine modelin bildiğini
  kullanmak" — sunumda güçlü bir başlık.
- **Model boyutu:** LoRA adaptörü 2.4 MB, tam fine-tuning 440 MB olurdu.
  Dağıtım/versiyonlama avantajı somut bir kazanım.
- **Aksan dayanıklılığı — "ölç, teşhis et, çöz, doğrula" döngüsünün tam örneği:**
  stil bazlı ölçüm gürültülüydü (güven aralıkları örtüşüyordu), müdahaleli
  nedensel test net cevap verdi (−6.4 puan), tokenizer mekanizmayı gösterdi
  (`asansör` 1 parça / `asansor` 3 parça), veri çoğaltma çözdü (−1.16 puan),
  aynı test doğruladı. Sunumda tek slaytta anlatılabilecek eksiksiz bir
  mühendislik hikâyesi.
- **"Otomatik tespit her zaman mümkün değil" dersi:** yabancı kelime tespiti
  için üç yöntem denendi, ikisi başarısız oldu (bigram sözlüğü 355 yanlış
  alarm; tokenizer parça sayısı yabancı kelimeyi ASCII Türkçe'den ayıramadı).
  Dar ve kesin bir kural, geniş ve gürültülü olandan iyidir — ve elle okuma
  hâlâ tek kapsamlı yöntem. Otomatik araçların sınırını gösteren dürüst bir
  bölüm, rapora olgunluk katar.
