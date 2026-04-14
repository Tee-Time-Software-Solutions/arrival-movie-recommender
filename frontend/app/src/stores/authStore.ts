import { create } from "zustand";
import type { User } from "firebase/auth";

interface AuthState {
  user: User | null;
  loading: boolean;
  token: string | null;
  firebaseUid: string | null;
  setUser: (user: User | null) => void;
  setLoading: (loading: boolean) => void;
  setToken: (token: string | null) => void;
  setFirebaseUid: (uid: string | null) => void;
  clear: () => void;
}

export const useAuthStore = create<AuthState>((set) => ({
  user: null,
  loading: true,
  token: null,
  firebaseUid: null,
  setUser: (user) => set({ user }),
  setLoading: (loading) => set({ loading }),
  setToken: (token) => set({ token }),
  setFirebaseUid: (uid) => set({ firebaseUid: uid }),
  clear: () => set({ user: null, token: null, firebaseUid: null, loading: false }),
}));
