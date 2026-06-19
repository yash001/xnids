import { NavLink, Outlet } from "react-router-dom";
import "./Shell.css";

const NAV_ITEMS = [
  { to: "/", label: "Dashboard", glyph: "◈" },
  { to: "/attacks", label: "Attack Screen", glyph: "⚡" },
  { to: "/defense", label: "Defence Screen", glyph: "▣" },
  { to: "/models", label: "Model Lab", glyph: "∿" },
];

export default function Shell() {
  return (
    <div className="shell">
      <aside className="shell__rail">
        <div className="shell__brand">
          <div className="shell__brand-mark">x</div>
          <div>
            <div className="shell__brand-name">SIGNAL</div>
            <div className="shell__brand-sub mono">xNIDS console</div>
          </div>
        </div>

        <nav className="shell__nav">
          {NAV_ITEMS.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) =>
                "shell__nav-item" + (isActive ? " is-active" : "")
              }
            >
              <span className="shell__nav-glyph">{item.glyph}</span>
              {item.label}
            </NavLink>
          ))}
        </nav>

        <div className="shell__footer">
          <div className="eyebrow">Based on</div>
          <div className="shell__paper">
            xNIDS — USENIX Security '23<br />Wei, Li, Zhao, Hu
          </div>
        </div>
      </aside>

      <main className="shell__main">
        <Outlet />
      </main>
    </div>
  );
}
