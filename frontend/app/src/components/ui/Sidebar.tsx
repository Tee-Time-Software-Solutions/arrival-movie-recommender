import { useLocation, useNavigate } from "react-router-dom";
import { IoHome, IoChatbubble, IoLogOutOutline, IoPersonCircleOutline } from "react-icons/io5";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";

const tabs = [
  { path: "/", icon: IoHome, label: "Discover" },
  { path: "/chat", icon: IoChatbubble, label: "Chat" },
  { path: "/profile", icon: IoPersonCircleOutline, label: "Profile" },
] as const;

interface SidebarProps {
  collapsed: boolean;
  onMouseEnter: () => void;
  onMouseLeave: () => void;
}

export function Sidebar({ collapsed, onMouseEnter, onMouseLeave }: SidebarProps) {
  const location = useLocation();
  const navigate = useNavigate();
  const clear = useAuthStore((s) => s.clear);

  const handleLogout = () => {
    clear();
    navigate("/landing");
  };

  return (
    <nav
      onMouseEnter={onMouseEnter}
      onMouseLeave={onMouseLeave}
      className={cn(
        "fixed left-0 top-0 z-50 hidden h-screen flex-col border-r border-border bg-background transition-[width] duration-300 md:flex",
        collapsed ? "w-[72px]" : "w-[240px]",
      )}
    >

      {/* Navigation */}
      <div className="flex flex-1 flex-col justify-center gap-1 px-3">
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
                  ? "bg-primary/10 font-semibold text-primary"
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

      {/* Logout */}
      <div className="px-3 pb-6">
        <button
          onClick={handleLogout}
          title={collapsed ? "Log out" : undefined}
          className={cn(
            "group relative flex w-full items-center rounded-lg py-3 text-sm text-muted-foreground transition-colors hover:bg-red-500/10 hover:text-red-500",
            collapsed ? "justify-center px-0" : "gap-4 px-3",
          )}
        >
          <IoLogOutOutline className="h-6 w-6 shrink-0" />
          {!collapsed && <span>Log out</span>}

          {collapsed && (
            <span className="pointer-events-none absolute left-full ml-2 hidden whitespace-nowrap rounded-md bg-foreground px-2 py-1 text-xs text-background group-hover:block">
              Log out
            </span>
          )}
        </button>
      </div>
    </nav>
  );
}
