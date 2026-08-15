export default function HomePage({ onChat }) {
  return (
    <main className="app home-page">
      <div className="orb" />
      <header>
        <div className="brand-mark">
          <img src="/icon.png" alt="" />
        </div>
        <div className="brand">
          <span>app</span>
        </div>
        <div className="header-status">
          <i /> LOCAL KNOWLEDGE ENGINE
        </div>
      </header>

      <section className="hero">
        <div className="eyebrow">PRIVATE DOCUMENT INTELLIGENCE</div>
        <h1>
          RAW DATA.
          <br />
          <em>PURE INTELLIGENCE.</em>
        </h1>
        <p>
          A calm interface for retrieval-augmented answers, grounded in the
          documents you upload.
        </p>
        <button className="home-cta" type="button" onClick={onChat}>
          Chat
          <span aria-hidden="true">→</span>
        </button>
      </section>
    </main>
  );
}
