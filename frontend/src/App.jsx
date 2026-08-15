import { useEffect, useState } from "react";
import HomePage from "./pages/HomePage";
import ChatPage from "./pages/ChatPage";

function normalizePath(path) {
  if (path === "/" || path === "") return "/home";
  return path.replace(/\/+$/, "") || "/home";
}

function NotFound({ onHome }) {
  return (
    <main className="not-found">
      <div className="astronaut big">◉</div>
      <div className="code">404</div>
      <p>This route drifted out of orbit.</p>
      <button className="primary" onClick={onHome}>
        Back to home
      </button>
    </main>
  );
}

export default function App() {
  const [route, setRoute] = useState(() =>
    normalizePath(window.location.pathname),
  );

  const navigate = (path) => {
    const next = normalizePath(path);
    if (next !== window.location.pathname) {
      history.pushState({}, "", next);
    } else if (window.location.pathname === "/" && next === "/home") {
      history.replaceState({}, "", "/home");
    }
    setRoute(next);
  };

  useEffect(() => {
    if (window.location.pathname === "/" || window.location.pathname === "") {
      history.replaceState({}, "", "/home");
      setRoute("/home");
    }

    const onPop = () => setRoute(normalizePath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  if (route === "/home") {
    return <HomePage onChat={() => navigate("/chat")} />;
  }

  if (route === "/chat") {
    return <ChatPage onHome={() => navigate("/home")} />;
  }

  return <NotFound onHome={() => navigate("/home")} />;
}
