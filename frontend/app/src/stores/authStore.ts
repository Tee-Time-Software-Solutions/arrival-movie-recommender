import { create } from "zustand";
import type { User } from "firebase/auth";

interface AuthState {
  user: User | null;
  loading: boolean;
  token: string | null;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setToken: (token: string | null) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  token: null,
  setUser: (user) => set({ user }),
  setLoading: (loading) => set({ loading }),
  setToken: (token) => set({ token }),
  clear: () => set({ user: null, token: null, loading: false }),
}));
