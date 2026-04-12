import { Navigate, useLocation } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading, needsOnboarding } = useAuthStore();
  const location = useLocation();

  if (loading) return null;

  if (!user) return <Navigate to="/landing" replace />;

  // Force onboarding if not completed (unless already on /onboarding)
  if (needsOnboarding && location.pathname !== "/onboarding") {
    return <Navigate to="/onboarding" replace />;
  }

  return <>{children}</>;
}
