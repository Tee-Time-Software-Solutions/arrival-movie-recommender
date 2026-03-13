import { useEffect } from "react";
import {
  onAuthStateChanged,
  signInWithPopup,
  signInWithEmailAndPassword,
  createUserWithEmailAndPassword,
  signOut as firebaseSignOut,
} from "firebase/auth";
import { auth, googleProvider } from "@/lib/firebase";
import { useAuthStore } from "@/stores/authStore";
import { registerUser } from "@/services/api/user";

export function useAuth() {
  const { user, loading, setUser, setLoading, setToken, setFirebaseUid, clear } =
    useAuthStore();

  useEffect(() => {
    const unsubscribe = onAuthStateChanged(auth, async (firebaseUser) => {
      if (firebaseUser) {
        const token = await firebaseUser.getIdToken();
        setUser(firebaseUser);
        setToken(token);
        setFirebaseUid(firebaseUser.uid);
      } else {
        setUser(null);
        setToken(null);
        setFirebaseUid(null);
      }
      setLoading(false);
    });

    return unsubscribe;
  }, [setUser, setLoading, setToken, setFirebaseUid]);

  const signInWithGoogle = async () => {
    const result = await signInWithPopup(auth, googleProvider);
    await tryRegisterUser(
      result.user.uid,
      result.user.email || "",
      result.user.photoURL || "",
    );
  };

  const signInWithEmail = async (email: string, password: string) => {
    await signInWithEmailAndPassword(auth, email, password);
  };

  const signUpWithEmail = async (email: string, password: string) => {
    const result = await createUserWithEmailAndPassword(auth, email, password);
    await tryRegisterUser(result.user.uid, email, "");
  };

  const signOut = async () => {
    await firebaseSignOut(auth);
    clear();
  };

  return {
    user,
    loading,
    signInWithGoogle,
    signInWithEmail,
    signUpWithEmail,
    signOut,
  };
}

async function tryRegisterUser(
  firebaseUid: string,
  email: string,
  profileImageUrl: string,
) {
  try {
    await registerUser({
      firebase_uid: firebaseUid,
      email,
      profile_image_url: profileImageUrl,
    });
  } catch (err: unknown) {
    const status = (err as { response?: { status?: number } })?.response?.status;
    if (status === 409) {
      // User already registered — safe to ignore
      return;
    }
    console.error("Failed to register user in backend:", err);
  }
}
