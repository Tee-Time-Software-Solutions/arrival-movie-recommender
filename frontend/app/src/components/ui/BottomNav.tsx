import { useLocation, useNavigate } from "react-router-dom";
import { IoHome, IoChatbubble, IoLogOutOutline, IoPersonCircleOutline } from "react-icons/io5";
import { cn } from "@/lib/utils";
import { useAuthStore } from "@/stores/authStore";

const tabs = [
  { path: "/", icon: IoHome, label: "Discover" },
  { path: "/chat", icon: IoChatbubble, label: "Chat" },
  { path: "/profile", icon: IoPersonCircleOutline, label: "Profile" },
] as const;

export function BottomNav() {
  const location = useLocation();
  const navigate = useNavigate();
  const clear = useAuthStore((s) => s.clear);

  const handleLogout = () => {
    clear();
    navigate("/landing");
  };

  return (
    <nav className="fixed bottom-0 left-0 right-0 z-50 border-t border-border bg-card/80 backdrop-blur-lg md:hidden">
      <div className="flex items-center justify-around py-2">
        {tabs.map(({ path, icon: Icon, label }) => {
          const isActive = location.pathname === path;
          return (
            <button
              key={path}
              onClick={() => navigate(path)}
              className={cn(
                "flex flex-col items-center gap-1 px-4 py-1 transition-colors",
                isActive
                  ? "text-primary"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              <Icon className="h-5 w-5" />
              <span className="text-xs">{label}</span>
            </button>
          );
        })}

        {/* Logout */}
        <button
          onClick={handleLogout}
          className="flex flex-col items-center gap-1 px-4 py-1 text-muted-foreground transition-colors hover:text-red-500"
        >
          <IoLogOutOutline className="h-5 w-5" />
          <span className="text-xs">Log out</span>
        </button>
      </div>
    </nav>
  );
}
