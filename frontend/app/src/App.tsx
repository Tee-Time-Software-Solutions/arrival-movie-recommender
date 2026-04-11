import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/ui/Layout";
import { ProtectedRoute } from "./components/ui/ProtectedRoute";
import { DiscoverPage } from "./app/Discover/DiscoverPage";
import { ChatPage } from "./app/Chat/ChatPage";
import { ProfilePage } from "./app/Profile/ProfilePage";
import { WatchlistPage } from "./app/Watchlist/WatchlistPage";
import { LandingPage } from "./app/Landing/LandingPage";
import { OnboardingPage } from "./app/Onboarding/OnboardingPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/landing" element={<LandingPage />} />
          <Route path="/onboarding" element={<ProtectedRoute><OnboardingPage /></ProtectedRoute>} />
          <Route path="/" element={<ProtectedRoute><DiscoverPage /></ProtectedRoute>} />
          <Route path="/chat" element={<ProtectedRoute><ChatPage /></ProtectedRoute>} />
          <Route path="/watchlist" element={<ProtectedRoute><WatchlistPage /></ProtectedRoute>} />
          <Route path="/profile" element={<ProtectedRoute><ProfilePage /></ProtectedRoute>} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
