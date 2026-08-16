export default function AppHeader({ onHome }) {
  return (
    <header className="app-header">
      <button
        className="brand-link"
        type="button"
        onClick={onHome}
        aria-label="Home"
      >
        <div className="brand-mark">
          <img src="/icon.png" alt="" />
        </div>
        <div className="brand">
          <span>app</span>
        </div>
      </button>
      <div className="header-status">
        <i /> LOCAL KNOWLEDGE ENGINE
      </div>
    </header>
  );
}
