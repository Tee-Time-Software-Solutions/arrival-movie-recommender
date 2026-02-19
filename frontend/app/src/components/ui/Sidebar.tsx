import { useLocation, useNavigate } from "react-router-dom";
import { MdExplore } from "react-icons/md";
import { IoChatbubbleEllipses } from "react-icons/io5";
import { FaUserCircle } from "react-icons/fa";
import {
  TbLayoutSidebarLeftCollapse,
  TbLayoutSidebarLeftExpand,
} from "react-icons/tb";
import { cn } from "@/lib/utils";

const tabs = [
  { path: "/", icon: MdExplore, label: "Discover" },
  { path: "/chat", icon: IoChatbubbleEllipses, label: "Chat" },
  { path: "/profile", icon: FaUserCircle, label: "Profile" },
] as const;

interface SidebarProps {
  collapsed: boolean;
  onToggle: () => void;
}

export function Sidebar({ collapsed, onToggle }: SidebarProps) {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <nav
      className={cn(
        "fixed left-0 top-0 z-50 hidden h-screen flex-col border-r border-border bg-background transition-[width] duration-300 md:flex",
        collapsed ? "w-[72px]" : "w-[240px]",
      )}
    >
      {/* Toggle + Logo */}
      <div className={cn("flex items-center py-6", collapsed ? "justify-center px-0" : "justify-between px-4")}>
        {!collapsed && (
          <h1 className="text-xl font-bold tracking-tight">Arrival</h1>
        )}
        <button
          onClick={onToggle}
          className="rounded-lg p-2 text-muted-foreground transition-colors hover:bg-accent/50 hover:text-foreground"
        >
          {collapsed ? (
            <TbLayoutSidebarLeftExpand className="h-6 w-6" />
          ) : (
            <TbLayoutSidebarLeftCollapse className="h-6 w-6" />
          )}
        </button>
      </div>

      {/* Navigation */}
      <div className="flex flex-1 flex-col gap-1 px-3">
        {tabs.map(({ path, icon: Icon, label }) => {
          const isActive = location.pathname === path;
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              title={collapsed ? label : undefined}
              className={cn(
                "group relative flex items-center rounded-lg py-3 text-sm transition-colors",
                collapsed ? "justify-center px-0" : "gap-4 px-3",
                isActive
                  ? "bg-accent font-semibold text-foreground"
                  : "text-muted-foreground hover:bg-accent/50 hover:text-foreground",
              )}
            >
              <Icon className="h-6 w-6 shrink-0" />
              {!collapsed && <span>{label}</span>}

              {/* Tooltip on hover when collapsed */}
              {collapsed && (
                <span className="pointer-events-none absolute left-full ml-2 hidden whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-xs text-background group-hover:block">
                  {label}
                </span>
              )}
            </button>
          );
        })}
      </div>
    </nav>
  );
}
