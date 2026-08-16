import { useEffect, useState } from "react";
import AppHeader from "./components/AppHeader";
import HomePage from "./pages/HomePage";
import ChatPage from "./pages/ChatPage";

function normalizePath(path) {
  if (path === "/" || path === "") return "/home";
  return path.replace(/\/+$/, "") || "/home";
}

function NotFound({ onHome }) {
  return (
    <div className="page not-found-page" key="404">
      <div className="astronaut big">◉</div>
      <div className="code">404</div>
      <p>This route drifted out of orbit.</p>
      <button className="primary" onClick={onHome}>
        Back to home
      </button>
    </div>
  );
}

export default function App() {
  const [route, setRoute] = useState("/home");

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
    history.replaceState({}, "", "/home");
    setRoute("/home");

    const onPop = () => setRoute(normalizePath(window.location.pathname));
    window.addEventListener("popstate", onPop);
    return () => window.removeEventListener("popstate", onPop);
  }, []);

  const isChat = route === "/chat";
  const isHome = route === "/home";

  let page = null;
  if (isHome) {
    page = <HomePage onChat={() => navigate("/chat")} />;
  } else if (isChat) {
    page = <ChatPage />;
  } else {
    page = <NotFound onHome={() => navigate("/home")} />;
  }

  return (
    <main
      className={[
        "app",
        isChat ? "chat-page" : "",
        isHome ? "home-page" : "",
      ]
        .filter(Boolean)
        .join(" ")}
    >
      <div className="orb" />
      <AppHeader onHome={() => navigate("/home")} />
      <div className="page-stage">{page}</div>
    </main>
  );
}
