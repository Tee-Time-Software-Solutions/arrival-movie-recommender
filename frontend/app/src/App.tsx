import { BrowserRouter, Routes, Route } from "react-router-dom";
import { Layout } from "./components/ui/Layout";
import { DiscoverPage } from "./app/Discover/DiscoverPage";
import { ChatPage } from "./app/Chat/ChatPage";
import { ProfilePage } from "./app/Profile/ProfilePage";
import { LandingPage } from "./app/Landing/LandingPage";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route element={<Layout />}>
          <Route path="/landing" element={<LandingPage />} />
          <Route path="/" element={<DiscoverPage />} />
          <Route path="/chat" element={<ChatPage />} />
          <Route path="/profile" element={<ProfilePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}

export default App;
