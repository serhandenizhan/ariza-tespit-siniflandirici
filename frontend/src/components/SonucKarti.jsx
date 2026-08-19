// Tahmin sonucu: birincil kategori, guven, uyarilar ve olasilik dagilimi.

const yuzde = (x) => `${(x * 100).toFixed(1)}%`;

export default function SonucKarti({ sonuc, kategoriler }) {
  // Backend olasiliklari sozluk olarak donduruyor; gorunum icin siralayip
  // her satira kendi kategori rengini (--k) tasiyoruz.
  const dagilim = Object.entries(sonuc.probabilities)
    .map(([anahtar, olasilik]) => ({
      anahtar,
      olasilik,
      ad: kategoriler[anahtar]?.label ?? anahtar,
      renk: kategoriler[anahtar]?.color ?? "#7a828f",
    }))
    .sort((a, b) => b.olasilik - a.olasilik);

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

      {/* manual_review tek bir operasyonel sinyal ama iki farkli sebepten
          tetiklenebiliyor. Operatore "neden bakmam gerekiyor" sorusunun
          cevabini vermek icin sebepleri ayri ayri yaziyoruz. */}
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
                  // Kademeli gecikme: barlar sirayla dolar, tek seferde
                  // hepsinin firlamasi yerine akici bir his veriyor
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
