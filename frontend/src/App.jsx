import { Routes, Route } from "react-router-dom";
import Shell from "./components/Shell";
import Dashboard from "./pages/Dashboard";
import AttackScreen from "./pages/AttackScreen";
import DefenceScreen from "./pages/DefenceScreen";
import ModelLab from "./pages/ModelLab";

export default function App() {
  return (
    <Routes>
      <Route element={<Shell />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/attacks" element={<AttackScreen />} />
        <Route path="/defense" element={<DefenceScreen />} />
        <Route path="/models" element={<ModelLab />} />
      </Route>
    </Routes>
  );
}
