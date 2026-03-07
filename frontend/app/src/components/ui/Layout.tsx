import { useState } from "react";
import { Outlet, useLocation } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { BottomNav } from "./BottomNav";

const HIDE_SHELL_ROUTES = ["/landing", "/auth"];

export function Layout() {
  const location = useLocation();
  const [hovered, setHovered] = useState(false);
  const hideShell = HIDE_SHELL_ROUTES.includes(location.pathname);

  return (
    <div className="min-h-screen bg-background text-foreground">
      {!hideShell && (
        <>
          <Sidebar
            collapsed={!hovered}
            onMouseEnter={() => setHovered(true)}
            onMouseLeave={() => setHovered(false)}
          />
          <BottomNav />
        </>
      )}

      <div
        data-layout-content
        className="flex min-h-screen flex-col transition-[margin-left] duration-300"
        style={{ marginLeft: hideShell ? 0 : hovered ? 240 : 72 }}
      >
        <Outlet />
      </div>
    </div>
  );
}
