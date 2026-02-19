import { useLocation, useNavigate } from "react-router-dom";
import { MdExplore } from "react-icons/md";
import { IoChatbubbleEllipses } from "react-icons/io5";
import { FaUserCircle } from "react-icons/fa";
import { cn } from "@/lib/utils";

const tabs = [
  { path: "/", icon: MdExplore, label: "Discover" },
  { path: "/chat", icon: IoChatbubbleEllipses, label: "Chat" },
  { path: "/profile", icon: FaUserCircle, label: "Profile" },
] as const;

export function BottomNav() {
  const location = useLocation();
  const navigate = useNavigate();

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
      </div>
    </nav>
  );
}
