import { Route, Routes } from "react-router-dom";

import { Layout } from "./components/Layout";
import { ExperimentPage } from "./pages/ExperimentPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { NewExperimentPage } from "./pages/NewExperimentPage";

export function App() {
  return (
    <Layout>
      <Routes>
        <Route path="/" element={<ExperimentsPage />} />
        <Route path="/new" element={<NewExperimentPage />} />
        <Route path="/experiments/:id" element={<ExperimentPage />} />
      </Routes>
    </Layout>
  );
}
