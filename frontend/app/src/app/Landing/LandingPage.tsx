import { useNavigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";
import { useEffect } from "react";
import { ArrowRight, ArrowLeft, ArrowUp, ArrowDown } from "lucide-react";

export function LandingPage() {
  const navigate = useNavigate();
  const { user, loading } = useAuthStore();

  useEffect(() => {
    if (!loading && user) {
      navigate("/discover", { replace: true });
    }
  }, [user, loading, navigate]);

  return (
    <div className="flex min-h-screen flex-col items-center justify-center px-6">
      <div className="w-full max-w-md text-center">
        <h1 className="mb-2 text-5xl font-bold tracking-tight">Arrival</h1>
        <p className="mb-10 text-lg text-muted-foreground">
          Discover movies you'll love, one swipe at a time.
        </p>

        <div className="mb-10 grid grid-cols-2 gap-4 text-sm">
          <div className="flex items-center gap-3 rounded-xl bg-card p-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-green-500/10">
              <ArrowRight className="h-5 w-5 text-green-500" />
            </div>
            <span className="text-left text-muted-foreground">
              Swipe right to <span className="text-foreground font-medium">like</span>
            </span>
          </div>
          <div className="flex items-center gap-3 rounded-xl bg-card p-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-red-500/10">
              <ArrowLeft className="h-5 w-5 text-red-500" />
            </div>
            <span className="text-left text-muted-foreground">
              Swipe left to <span className="text-foreground font-medium">pass</span>
            </span>
          </div>
          <div className="flex items-center gap-3 rounded-xl bg-card p-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-blue-500/10">
              <ArrowUp className="h-5 w-5 text-blue-500" />
            </div>
            <span className="text-left text-muted-foreground">
              Swipe up for <span className="text-foreground font-medium">details</span>
            </span>
          </div>
          <div className="flex items-center gap-3 rounded-xl bg-card p-3">
            <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-yellow-500/10">
              <ArrowDown className="h-5 w-5 text-yellow-500" />
            </div>
            <span className="text-left text-muted-foreground">
              Swipe down to <span className="text-foreground font-medium">rate</span>
            </span>
          </div>
        </div>

        <div className="flex flex-col gap-3">
          <button
            onClick={() => navigate("/auth")}
            className="w-full rounded-xl bg-primary px-6 py-3 text-sm font-semibold text-primary-foreground transition hover:bg-primary/90"
          >
            Get Started
          </button>
          <button
            onClick={() => navigate("/auth")}
            className="w-full rounded-xl bg-secondary px-6 py-3 text-sm font-semibold text-secondary-foreground transition hover:bg-secondary/80"
          >
            I already have an account
          </button>
        </div>
      </div>
    </div>
  );
}
