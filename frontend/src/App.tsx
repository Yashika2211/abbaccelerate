import { Navigate, Route, Routes } from "react-router-dom";
import Layout from "./components/Layout";
import CostStudio from "./pages/CostStudio";
import Data from "./pages/Data";
import Explain from "./pages/Explain";
import Leaderboard from "./pages/Leaderboard";
import Run from "./pages/Run";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Layout />}>
        <Route index element={<Data />} />
        <Route path="cost" element={<CostStudio />} />
        <Route path="run" element={<Run />} />
        <Route path="leaderboard" element={<Leaderboard />} />
        <Route path="explain" element={<Explain />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
