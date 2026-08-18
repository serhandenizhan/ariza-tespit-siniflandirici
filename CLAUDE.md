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
│   │   ├── seed.jsonl              # few-shot yemi, ~96 kayit (bkz. "Veri Durumu")
│   │   ├── gold.jsonl              # bozulmamis test seti, ~79 kayit
│   │   ├── seed_v1_backup.jsonl    # Gemini ile ilk deneme (arsiv)
│   │   ├── gold_v1_backup.jsonl
│   │   ├── seed_v3_groq_fixed_backup.jsonl  # Groq/Llama3.3 duzeltilmis prompt (arsiv)
│   │   └── gold_v3_groq_fixed_backup.jsonl
│   ├── raw/                        # BOS -- Adim 2b'de Ollama ciktisi buraya
│   └── processed/                  # BOS -- Adim 3'te train/val/test buraya
├── model/                          # BOS -- egitilmis model buraya
├── src/
│   ├── __init__.py
│   ├── config.py                   # TEK dogruluk kaynagi, asagida detay
│   ├── generate_seed.py            # seed/gold ureten script (coklu saglayici)
│   ├── review.py                   # otomatik kalite triyaj araci
│   ├── apply_review.py             # elle onaylanan duzeltmeleri uygular
│   ├── check_openrouter_models.py  # OpenRouter canli model/fiyat listesi
│   ├── generate_data.py            # YOK -- Adim 2b'de yazilacak (cogaltma)
│   ├── preprocess.py                # YOK -- Adim 3'te yazilacak
│   ├── train.py                     # YOK -- Adim 4'te yazilacak
│   └── evaluate.py                  # YOK -- Adim 4'te yazilacak
├── backend/
│   └── main.py                      # YOK -- Adim 5'te yazilacak (FastAPI)
├── frontend/                        # YOK -- Adim 6'da yazilacak (React)
├── venv/
└── .env                             # GEMINI_API_KEY, GROQ_API_KEY,
                                      # OPENROUTER_API_KEY var; ANTHROPIC_API_KEY yok
```

`requirements.txt` henuz konsolide edilmedi -- su ana kadar `pip install`
ile tek tek kuruldu (python-dotenv, google-genai, anthropic, requests, groq,
openai). Adim 2b'ye gecmeden once `pip freeze > requirements.txt` ile
donduruleblir.

## config.py — Projenin Tek Doğruluk Kaynağı

Tüm diğer modüller (mevcut ve yazılacak olanlar) buradan import eder. Asla
kategori/stil/hiperparametre tanımı başka bir dosyada tekrarlanmaz.

### Kategori Taksonomisi (8 kategori)

**Ayrım ilkesi: kategori, bildirimin hangi BAKIM EKİBİNE yönlendirileceğini
belirtir. Arızanın nesnesi değil, sorumlusu belirleyicidir.**

| key | display | kapsam (özet) |
|---|---|---|
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
|---|---|---|
| `standart` | 8-18 kelime | Düzgün, kurallı tam cümle |
| `devrik` | 4-9 kelime | Acele yazılmış, kısa, eksiltili (özne/yüklem sırası bozuk olabilir) |
| `yazim_yanlisi` | 5-14 kelime | Türkçe karakter eksikliği, klavye hatası |
| `cok_kisa` | 3-6 kelime | Telgraf tarzı, sadece ekipman + belirti |

**Önemli tasarım kararı:** Türkçe aksan düşürme (`güvenlik`→`guvenlik`,
`Şişli`→`Sisli`) **sadece** `yazim_yanlisi` stiline özgü değil — gerçek
hayatta İngilizce klavyeyle hızlı yazan biri her stilde bunu yapabilir.
Bu davranış bir "hata" değil, modelin dayanıklı olması gereken doğal bir
varyasyon olarak kabul edildi (bkz. review.py bölümü).

### Hedef Veri Hacmi

- Seed: 12/kategori × 8 = 96 (few-shot yemi, Ollama'ya verilecek)
- Gold: 10/kategori × 8 = 80 (few-shot'ta **asla** kullanılmaz, saf test seti)
- Nihai eğitim verisi (çoğaltma sonrası): **200/kategori × 8 = 1600**

(Not: orijinal PDF taslağında 70/kategori × 6 kategori = 420 yazıyordu.
Kategori sayısı 6'dan 8'e, hedef hacim 70'ten 200'e çıkarıldı — rapor
güncellenmeli.)

### Eğitim Hiperparametreleri (henüz kullanılmadı, Adım 4 için hazır)

- Model: `dbmdz/bert-base-turkish-cased`
- PEFT/LoRA: r=16, alpha=32, dropout=0.1, target_modules=["query","value"]
- MAX_LENGTH=64, NUM_EPOCHS=5, BATCH_SIZE=16, LEARNING_RATE=2e-5
- Split: %80 train / %10 val / %10 test
- Başarı kriteri: accuracy ≥ 0.85, macro F1 ≥ 0.82, hiçbir sınıf F1 < 0.75

### Servis Ayarları (henüz kullanılmadı, Adım 5 için hazır)

- `CONFIDENCE_THRESHOLD = 0.60` — bu eşiğin altında kategori atanmaz,
  `low_confidence: true` döner, arayüzde "Düşük Güven: Manuel İnceleme
  Önerilir" gösterilir. **Bu değer henüz kalibre edilmedi** — evaluate.py
  yazıldıktan sonra gerçek güven dağılımına bakıp ayarlanacak.

## LLM Sağlayıcı Yolculuğu (önemli — rapora doğrudan girebilecek içerik)

Seed/gold üretimi için 4 farklı sağlayıcı denendi, sonuçlar `review.py` ile
ölçüldü. Bu, projenin metodoloji bölümüne güçlü bir katkı:

| Deneme | Sağlayıcı/Model | Seed işaretli | Gold işaretli | Not |
|---|---|---|---|---|
| v1 | Gemini (çeşitli modeller) | %18 | %32 | Kota/model erişim sorunları yüzünden terk edildi |
| v2 | Groq / llama-3.3-70b-versatile | %91 | %81 | Config'teki Türkçe metin ASCII'ydi (kök sebep) |
| v3 | Groq / llama-3.3-70b-versatile (Türkçe düzeltildi) | %72 | %69 | Prompt dili düzelince iyileşti ama model hâlâ çok-kısıtlı talimatlara uyamadı |
| **v4** | **OpenRouter / nvidia/nemotron-3-ultra-550b-a55b:free** | **%17** | **%9** | **Kazanan.** Ücretsiz, kart istemiyor, v1'i bile geçti |

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

## review.py — Kalite Kontrol Sistemi

Otomatik olarak şunları işaretler (elle bakılması gerekenler):
- `DUP` — birebir tekrar
- `BENZER` — yakın kopya (SequenceMatcher + Jaccard kelime-kümesi hibrit,
  1.0'da sınırlı)
- `UZUNLUK` — stilin beklediği kelime aralığı dışında
- `SIZINTI` — kategori adını çağrıştıran kelime, kategori başına %25 payını
  aşıyor

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

## ⚠️ Doğrulanmamış / Açık Noktalar (Claude Code önce bunları kontrol etsin)

1. **`apply_review.py` gerçekten çalıştırıldı mı, sonucu doğrulandı mı?**
   Son mesajda "çalıştır, sonra `review.py --only-flagged` ile doğrula"
   denildi ama çalıştırma çıktısı bu konuşmada paylaşılmadı. İlk iş: bunu
   çalıştırıp/doğrulayıp gerçek son hâli teyit et.
2. **Gold'da bir kategori eksik:** v4 sonucunda `Güvenlik / Emniyet`
   kategorisinde 9/10 kayıt vardı (10 değil). `resume` mantığı %80 eşiğini
   (8/10) geçtiği için otomatik tamamlamadı. İsteğe bağlı: `python -m
   src.generate_seed --only gold --provider openrouter` ile üstüne
   eklenebilir (mevcut kategoriler dokunulmadan kalır, sadece eksik
   kategori denenir çünkü 9 < 8 değil aslında -- yani bu otomatik
   tetiklenmez, elle `--force` ile o kategoriye özel bir tamamlama
   gerekebilir; script şu an kategori-özel force desteklemiyor, küçük bir
   ek gerekebilir).
3. **`SEED_PROVIDER` düzeltildi:** Bu md yazılırken fark edildi ki
   `config.py`'de hâlâ `"groq"` yazıyordu, gerçek kazanan `"openrouter"`
   idi. Bu dosyada düzeltildi, ama teyit et.
4. **qwen2.5:14b vs gemma2:9b karşılaştırması hiç yapılmadı.** Ollama'da
   qwen2.5:14b kurulu ve varsayılan seçildi (RAM yeterli olduğu için), ama
   planlanan yan-yana kalite testi atlandı çünkü Nemotron sonuçları zaten
   tatmin ediciydi. Adım 2b'ye başlarken hâlâ gerekli mi karar verilmeli.
5. **Açık karar: Adım 2b'de (çoğaltma) hangi model kullanılacak?**
   Orijinal plan "seed/gold güçlü modelle, çoğaltma Ollama'yla (ücretsiz,
   yerel)" şeklindeydi. Ama OpenRouter/Nemotron 3 Ultra bu kadar iyi
   sonuç verdiği için, 1600 cümlelik çoğaltmayı da OpenRouter üzerinden
   yapmak (aynı ücretsiz kota dahilinde, ~50 istek/gün limiti göz önünde
   bulundurularak batch'lenerek) kalite açısından daha iyi olabilir. Bu
   konuşulmadı, karar verilmesi lazım. Ollama'nın avantajı: tamamen
   yerel, kota derdi yok, MacBook'un fanı hariç bedel yok.

## Yol Haritası — Kalan Adımlar

**Adım 2b — Çoğaltma (Amplification):** `src/generate_data.py` yazılacak.
Few-shot olarak `seed.jsonl` kullanılacak (gold ASLA kullanılmaz). Model
seçimi yukarıdaki açık karara bağlı (Ollama qwen2.5:14b veya OpenRouter
Nemotron). `SLOT_VALUES` (istasyon/konum/zaman/aciliyet listeleri
`config.py`'de hazır) her çağrıda rastgele enjekte edilerek çeşitlilik
prompt seviyesinde zorlanacak. Hedef: 200/kategori × 8 = 1600. `review.py`
bu çıktı için de kullanılmalı (kategori/stil argümanları zaten genel).

**Adım 3 — Ön İşleme:** `src/preprocess.py` yazılacak.
- **Kritik:** Split'ten önce near-duplicate kümeleme yapılmalı (çoğaltılmış
  veri birbirine çok benzeyecek, rastgele split yaparsa aynı cümlenin
  varyasyonu hem train hem test'e düşer, sahte yüksek doğruluk çıkar).
  `review.py`'deki `similarity()` fonksiyonu bu iş için yeniden
  kullanılabilir.
  - %80/%10/%10 train/val/test split
- Gold seti asla train/val'e karışmaz, ayrı `gold_test.csv` olarak kalır
  (iki ayrı test metriği raporlanacak: normal test + gold test — aradaki
  fark "sentetik veri ne kadar gerçekçi" sorusunun kanıtı olur, rapora
  güçlü bir savunma katar)

**Adım 4 — Model Eğitimi:** `src/train.py` + `src/evaluate.py` yazılacak.
BERT + LoRA fine-tuning, HuggingFace Trainer, scikit-learn ile
accuracy/macro-F1/confusion matrix. Hedefler `config.py`'de tanımlı
(TARGET_ACCURACY, TARGET_MACRO_F1, MIN_PER_CLASS_F1). Apple Silicon'da MPS
backend kullanılacak, LoRA/PEFT ile bilinen dtype/uyumluluk sorunları
çıkabilir, dikkatli debug gerekebilir.

**Adım 5 — Backend:** `backend/main.py`, FastAPI. `/predict` endpoint,
`CONFIDENCE_THRESHOLD` (0.60) altında `low_confidence: true` dönecek.
Model Python tarafında yüklü tutulacak, REST API olarak servis edilecek.

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

## Git İş Akışı Kuralı (Claude Code bunu her adımda uygulasın)

Bu proje adım adım (Adım 2b, 3, 4, 5, 6...) ilerliyor. Her adım için şu sıra
**kesinlikle** izlenir:

1. **Kodu yaz/düzenle.**
2. **Çalıştır ve test et.** Sadece kod yazmak yeterli değildir — script'i
   gerçekten çalıştır, çıktısını gör, hata varsa düzelt, tekrar çalıştır.
   "Kodu düzenledim" ile "denedim ve çalıştığını doğruladım" farklı
   şeylerdir; sadece ikincisi bir adımı tamamlanmış sayar.
3. **Adım gerçekten çalıştığı doğrulanınca**, bana kısaca ne yapıldığını ve
   test sonucunu özetle, **git'e commit + push için onay iste.**
4. **Benden açık onay gelmeden asla `git push` yapma.** "Onaylıyorum",
   "push'la", "evet" gibi net bir cevap bekle. Onay gelmeden bir sonraki
   adıma da geçme — sırayla ilerle.
5. Onay gelince o adımı **kendi başına bir commit** olarak işle (önceki
   adımlarla birleştirme) ve push et. Commit mesajı hangi adım olduğunu
   ve ne yapıldığını açıkça belirtsin (ör. `Adım 2b: Ollama ile çoğaltma
   scripti (generate_data.py) + 1600 örnek üretimi`).
6. Böylece git geçmişinde proje adım adım, her biri çalıştığı doğrulanmış
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
