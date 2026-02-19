import { useState } from "react";
import { Outlet } from "react-router-dom";
import { Sidebar } from "./Sidebar";
import { BottomNav } from "./BottomNav";

export function Layout() {
  const [collapsed, setCollapsed] = useState(false);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <Sidebar collapsed={collapsed} onToggle={() => setCollapsed((c) => !c)} />
      <BottomNav />

      <div
        data-layout-content
        className="flex min-h-screen flex-col transition-[margin-left] duration-300"
        style={{ marginLeft: collapsed ? 72 : 240 }}
      >
        <Outlet />
      </div>
    </div>
  );
}
