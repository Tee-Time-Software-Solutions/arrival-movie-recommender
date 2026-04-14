import { Navigate } from "react-router-dom";
import { useAuthStore } from "@/stores/authStore";

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { user, loading } = useAuthStore();

  if (loading) return null;

  if (!user) return <Navigate to="/landing" replace />;

  return <>{children}</>;
}
