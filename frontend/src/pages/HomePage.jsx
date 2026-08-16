export default function HomePage({ onChat }) {
  return (
    <div className="page home-page" key="home">
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
    </div>
  );
}
