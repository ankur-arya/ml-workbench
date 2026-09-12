import { NavLink } from "react-router-dom";
import type { Bootstrap } from "../types";
import { useState } from "react";

export function Layout({
  boot,
  onRefresh,
  children,
}: {
  boot: Bootstrap | null;
  onRefresh: () => void;
  children: React.ReactNode;
}) {
  const [theme, setTheme] = useState(document.documentElement.dataset.theme || "dark");

  function toggleTheme() {
    const next = theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = next;
    localStorage.setItem("wb-theme", next);
    setTheme(next);
  }

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">
            Work<em>bench</em>
          </div>
          <div className="brand-sub">Evaluate · Compare · Promote</div>
        </div>
        <nav className="nav">
          <NavLink to="/" end>
            Experiments <span className="count">{boot?.stats.experiments ?? 0}</span>
          </NavLink>
          <NavLink to="/new">New experiment</NavLink>
          <NavLink to="/registry">
            Registry <span className="count">{boot?.stats.production_models ?? 0} prod</span>
          </NavLink>
          <NavLink to="/datasets">
            Datasets <span className="count">{boot?.stats.datasets ?? 0}</span>
          </NavLink>
        </nav>
        <div className="sidebar-foot">
          <div className="pill">
            <span className="dot" />
            Local · no Docker
          </div>
          <button className="btn ghost" onClick={toggleTheme}>
            {theme === "dark" ? "Light theme" : "Dark theme"}
          </button>
          <button className="btn ghost" onClick={onRefresh}>
            Refresh
          </button>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}
