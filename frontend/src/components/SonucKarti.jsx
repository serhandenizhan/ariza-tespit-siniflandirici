import { useState } from "react";
import { tahminiDogrula } from "../api";

// Tahmin sonucu: birincil kategori, guven, uyarilar, benzerlik ve olasilik dagilimi.

const yuzde = (x) => `${(x * 100).toFixed(1)}%`;

// Yapisal alanlar (Adim 7/9) icin goruntu adi -- backend anahtar Ingilizce
// donuyor (extract.py sozlesmesi), arayuzde Turkce etiket gosteriliyor.
const VARLIK_ETIKETLERI = [
  ["line", "Hat"],
  ["station", "İstasyon"],
  ["location", "Konum"],
  ["equipment", "Ekipman"],
  ["symptom", "Belirti"],
  ["root_cause", "Kök Sebep"],
];

export default function SonucKarti({ sonuc, kategoriler, onDogrulandi }) {
  const [onayDurumu, setOnayDurumu] = useState(null); // null | "dogru" | "yanlis" | "duzeltiliyor"
  const [gonderiliyor, setGonderiliyor] = useState(false);

  // Dagilimi olasiliga gore sirala; backend sozluk donduruyor.
  const dagilim = Object.entries(sonuc.probabilities)
    .map(([anahtar, olasilik]) => ({
      anahtar,
      olasilik,
      ad: kategoriler[anahtar]?.label ?? anahtar,
      renk: kategoriler[anahtar]?.color ?? "#7a828f",
    }))
    .sort((a, b) => b.olasilik - a.olasilik);

  const varliklar = VARLIK_ETIKETLERI.filter(([anahtar]) => sonuc[anahtar]);
  const VARLIK_ETIKET_SOZLUGU = Object.fromEntries(VARLIK_ETIKETLERI);
  const eksikBilgiEtiketli = sonuc.missing_information.map(
    (anahtar) => VARLIK_ETIKET_SOZLUGU[anahtar] ?? anahtar
  );

  const dogrulanabilir = sonuc.log_id > 0 && onayDurumu === null;

  async function gonder(dogru, duzeltilmisKategori = null) {
    setGonderiliyor(true);
    try {
      await tahminiDogrula(sonuc.log_id, dogru, duzeltilmisKategori);
      setOnayDurumu(dogru ? "dogru" : "yanlis");
      onDogrulandi?.();
    } catch {
      /* sessizce yut -- onay ikincil bir eylem, akisi kesmemeli */
    } finally {
      setGonderiliyor(false);
    }
  }

  return (
    <section className="sonuc" style={{ "--vurgu": sonuc.color }} aria-live="polite">
      <header className="sonuc-baslik">
        <span className="rozet" style={{ "--k": sonuc.color }}>
          <span className="nokta" />
          {sonuc.label}
        </span>

        {sonuc.secondary_category && (
          <>
            <span className="arti" aria-hidden="true">+</span>
            <span
              className="rozet rozet-ikincil"
              style={{ "--k": kategoriler[sonuc.secondary_category]?.color }}
              title="Sınırda bildirim — ikinci ekip de değerlendirmeli"
            >
              {sonuc.secondary_label} · {yuzde(sonuc.secondary_confidence)}
            </span>
          </>
        )}
      </header>

      <div className="meta-satir">
        <span className="meta-cip meta-intent" title="Kullanıcının amacı">
          {sonuc.intent_label}
        </span>
        <span
          className="meta-cip meta-oncelik"
          style={{ "--k": sonuc.priority_color }}
          title={
            sonuc.priority_rule
              ? `Kural katmanı tetiklendi: ${sonuc.priority_rule}`
              : `Model güveni: ${yuzde(sonuc.priority_confidence)}`
          }
        >
          <span className="nokta" />
          {sonuc.priority_label}
          {sonuc.priority_rule && <span className="meta-kural">kural</span>}
        </span>
      </div>

      <div className="guven">
        <div className="guven-ust">
          <span className="guven-etiket">Güven</span>
          <span className="guven-deger">{yuzde(sonuc.confidence)}</span>
        </div>
        <div className="ray">
          <div
            className="dolgu"
            style={{ width: yuzde(sonuc.confidence), "--k": sonuc.color }}
          />
        </div>
      </div>

      {varliklar.length > 0 && (
        <div className="varliklar">
          <h3 className="bolum-baslik">Yapısal Bilgiler</h3>
          <dl className="varlik-liste">
            {varliklar.map(([anahtar, etiket]) => (
              <div className="varlik-satir" key={anahtar}>
                <dt>{etiket}</dt>
                <dd>{sonuc[anahtar]}</dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {sonuc.evidence.length > 0 && (
        <div className="kanit">
          <h3 className="bolum-baslik">
            Kanıt
            <span className="benzer-sayi">
              {" "}
              — modelin en çok dikkate aldığı kelimeler
            </span>
          </h3>
          <div className="kanit-liste">
            {sonuc.evidence.map((kelime, i) => (
              <span className="kanit-cip" key={`${kelime}-${i}`}>
                {kelime}
              </span>
            ))}
          </div>
        </div>
      )}

      {sonuc.missing_information.length > 0 && (
        <div className="bilgi">
          <div className="bilgi-baslik">
            <span aria-hidden="true">ⓘ</span> Eksik Bilgi
          </div>
          <p>
            İş emri için şu bilgiler bildirimde yer almıyor:{" "}
            {eksikBilgiEtiketli.join(", ")}.
          </p>
        </div>
      )}

      {sonuc.possible_duplicate && sonuc.duplicate_of && (
        <div className="uyari">
          <div className="uyari-baslik">
            <span aria-hidden="true">⟳</span> Olası Tekrar Bildirim
          </div>
          <ul>
            <li>
              Aynı arıza son 15 dakikada <b>{sonuc.duplicate_of.sayi}</b> kez
              bildirilmiş. İlk bildirim: “{sonuc.duplicate_of.ilk_metin}”
            </li>
          </ul>
        </div>
      )}

      {sonuc.manual_review && (
        <div className="uyari">
          <div className="uyari-baslik">
            <span aria-hidden="true">⚠</span> Manuel İnceleme Önerilir
          </div>
          <ul>
            {sonuc.low_confidence && (
              <li>
                Güven eşiğin altında (<b>{yuzde(sonuc.confidence)}</b>) — model
                bu bildirimden emin değil.
              </li>
            )}
            {sonuc.secondary_category && (
              <li>
                Model <b>{sonuc.label}</b> ile <b>{sonuc.secondary_label}</b>{" "}
                arasında kararsız (fark <b>{yuzde(sonuc.margin)}</b>). Bu
                bildirim gerçekten iki ekibi birden ilgilendiriyor olabilir.
              </li>
            )}
          </ul>
        </div>
      )}

      {/* Onay: kullanicinin isaretledigi kayitlar SADECE bunlar /logs/export
          ile disari alinip elle egitime katilabilir -- bkz. backend/src/db.py */}
      <div className="onay">
        {onayDurumu === null && (
          <>
            <span className="onay-soru">Bu tahmin doğru mu?</span>
            <button
              className="onay-btn onay-dogru"
              disabled={!dogrulanabilir || gonderiliyor}
              onClick={() => gonder(true)}
            >
              ✓ Doğru
            </button>
            <button
              className="onay-btn onay-yanlis"
              disabled={!dogrulanabilir || gonderiliyor}
              onClick={() => setOnayDurumu("duzeltiliyor")}
            >
              ✕ Yanlış
            </button>
          </>
        )}
        {onayDurumu === "duzeltiliyor" && (
          <>
            <span className="onay-soru">Doğrusu:</span>
            <select
              className="onay-secim"
              disabled={gonderiliyor}
              defaultValue=""
              onChange={(e) => e.target.value && gonder(false, e.target.value)}
            >
              <option value="" disabled>seçin…</option>
              {Object.values(kategoriler)
                .filter((k) => k.category !== sonuc.category)
                .map((k) => (
                  <option key={k.category} value={k.category}>{k.label}</option>
                ))}
            </select>
            <button
              className="onay-btn onay-vazgec"
              onClick={() => gonder(false)}
              disabled={gonderiliyor}
              title="Kategori belirtmeden sadece 'yanlış' olarak işaretle"
            >
              atla
            </button>
          </>
        )}
        {onayDurumu === "dogru" && <span className="onay-tesekkur">✓ Teşekkürler, kaydedildi.</span>}
        {onayDurumu === "yanlis" && <span className="onay-tesekkur">✓ Düzeltme kaydedildi.</span>}
        {sonuc.log_id <= 0 && onayDurumu === null && (
          <span className="onay-soru onay-devre-disi">
            (bu istek tekrar olduğu için loglanmadı)
          </span>
        )}
      </div>

      {sonuc.similar.total_found > 0 && (
        <div className="benzer">
          <h3 className="bolum-baslik">
            Benzer Kayıtlar
            <span className="benzer-sayi">
              {" "}
              — {sonuc.similar.total_found} kayıt bulundu
              {sonuc.similar.shown < sonuc.similar.total_found &&
                ` (ilk ${sonuc.similar.shown} gösteriliyor)`}
            </span>
          </h3>
          {sonuc.similar.distribution.map((d) => (
            <div className="dagilim-satir" key={d.category}>
              <span className="dagilim-ad">{d.label}</span>
              <div className="ray ince">
                <div
                  className="dolgu"
                  style={{ width: yuzde(d.ratio), "--k": d.color }}
                />
              </div>
              <span className="dagilim-deger">
                {d.count} · {yuzde(d.ratio)}
              </span>
            </div>
          ))}
        </div>
      )}

      <div className="dagilim">
        <h3 className="bolum-baslik">Tüm kategoriler</h3>
        {dagilim.map((d, i) => (
          <div
            className={`dagilim-satir${i === 0 ? " birincil" : ""}`}
            key={d.anahtar}
          >
            <span className="dagilim-ad">{d.ad}</span>
            <div className="ray ince">
              <div
                className="dolgu"
                style={{
                  width: yuzde(d.olasilik),
                  "--k": d.renk,
                  animationDelay: `${60 + i * 45}ms`,
                }}
              />
            </div>
            <span className="dagilim-deger">{yuzde(d.olasilik)}</span>
          </div>
        ))}
      </div>

      <footer className="sonuc-alt">
        <span>Yanıt süresi {sonuc.response_time_ms.toFixed(1)} ms</span>
        <span>top-2 farkı {yuzde(sonuc.margin)}</span>
      </footer>
    </section>
  );
}
