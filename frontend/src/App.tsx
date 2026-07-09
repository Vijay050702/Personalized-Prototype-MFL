/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { Dashboard } from './pages/Dashboard';
import { Clients } from './pages/Clients';
import { Datasets } from './pages/Datasets';
import { Training } from './pages/Training';
import { Prototypes } from './pages/Prototypes';
import { KnowledgeTransfer } from './pages/KnowledgeTransfer';
import { Similarity } from './pages/Similarity';
import { Evaluation } from './pages/Evaluation';
import { Experiments } from './pages/Experiments';
import { Settings } from './pages/Settings';

export default function App() {
  return (
    <BrowserRouter>
      <MainLayout>
        <Routes>
          <Route path="/" element={<Dashboard />} />
          <Route path="/clients" element={<Clients />} />
          <Route path="/datasets" element={<Datasets />} />
          <Route path="/training" element={<Training />} />
          <Route path="/prototypes" element={<Prototypes />} />
          <Route path="/knowledge-transfer" element={<KnowledgeTransfer />} />
          <Route path="/similarity" element={<Similarity />} />
          <Route path="/evaluation" element={<Evaluation />} />
          <Route path="/experiments" element={<Experiments />} />
          <Route path="/settings" element={<Settings />} />
          <Route path="*" element={<div className="text-on-surface p-12 text-center">404 - Not Found</div>} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}
