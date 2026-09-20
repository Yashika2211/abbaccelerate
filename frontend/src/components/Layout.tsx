import { NavLink, Outlet } from "react-router-dom";
import { useEffect, useState } from "react";
import { api } from "../lib/api";
import { useSession } from "../lib/session";
import type { Health } from "../lib/types";
import { Dot } from "./ui";

const TABS = [
  { to: "/", label: "Data", end: true },
  { to: "/cost", label: "Cost Studio" },
  { to: "/run", label: "Run" },
  { to: "/leaderboard", label: "Leaderboard" },
];

export default function Layout() {
  const [health, setHealth] = useState<Health | null>(null);
  const [session] = useSession();

  useEffect(() => {
    let alive = true;
    const poll = () => api.health().then((h) => alive && setHealth(h)).catch(() => {});
    poll();
    const id = setInterval(poll, 8000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, []);

  return (
    <div className="flex min-h-full flex-col">
      <header className="border-edge bg-panel/60 border-b backdrop-blur">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center gap-x-6 gap-y-2 px-6 py-3">
          <div className="flex items-baseline gap-2">
            <span className="text-lg font-semibold tracking-tight">Kairos</span>
            <span className="text-muted hidden text-xs sm:inline">the right moment to act</span>
          </div>

          <nav className="flex gap-1">
            {TABS.map((tab) => (
              <NavLink
                key={tab.to}
                to={tab.to}
                end={tab.end}
                className={({ isActive }) =>
                  `rounded px-2.5 py-1 text-xs font-medium transition ${
                    isActive ? "bg-edge text-white" : "text-muted hover:text-white"
                  }`
                }
              >
                {tab.label}
              </NavLink>
            ))}
          </nav>

          <div className="ml-auto flex items-center gap-3 text-xs">
            {session.datasetName && (
              <span className="text-muted hidden md:inline">{session.datasetName}</span>
            )}
            {health ? (
              <>
                <Dot up={health.components.postgres === "up"} label="db" />
                <Dot up={health.components.mlflow === "up"} label="mlflow" />
                <Dot
                  up={health.components.llm === "configured"}
                  label={health.components.llm === "configured" ? "llm" : "templates"}
                />
              </>
            ) : (
              <Dot up={false} label="api" />
            )}
          </div>
        </div>
      </header>

      <main className="mx-auto w-full max-w-6xl flex-1 px-6 py-6">
        <Outlet />
      </main>

      <footer className="text-muted mx-auto w-full max-w-6xl px-6 pb-6 text-[11px]">
        Every figure on screen is computed from the loaded dataset. Nothing is mocked; where a
        value is unavailable the page says so.
      </footer>
    </div>
  );
}
