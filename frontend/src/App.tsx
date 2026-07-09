/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { MainLayout } from './components/layout/MainLayout';
import { ProtectedRoute } from './routes/ProtectedRoute';
import { LoginPage } from './components/auth/LoginPage';
import { LiveDashboard } from './components/realtime/LiveDashboard';
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
          <Route path="/login" element={<LoginPage />} />
          <Route path="/" element={<ProtectedRoute><Dashboard /></ProtectedRoute>} />
          <Route path="/live" element={<ProtectedRoute><LiveDashboard /></ProtectedRoute>} />
          <Route path="/clients" element={<ProtectedRoute><Clients /></ProtectedRoute>} />
          <Route path="/datasets" element={<ProtectedRoute><Datasets /></ProtectedRoute>} />
          <Route path="/training" element={<ProtectedRoute><Training /></ProtectedRoute>} />
          <Route path="/prototypes" element={<ProtectedRoute><Prototypes /></ProtectedRoute>} />
          <Route path="/knowledge-transfer" element={<ProtectedRoute><KnowledgeTransfer /></ProtectedRoute>} />
          <Route path="/similarity" element={<ProtectedRoute><Similarity /></ProtectedRoute>} />
          <Route path="/evaluation" element={<ProtectedRoute><Evaluation /></ProtectedRoute>} />
          <Route path="/experiments" element={<ProtectedRoute><Experiments /></ProtectedRoute>} />
          <Route path="/settings" element={<ProtectedRoute><Settings /></ProtectedRoute>} />
          <Route path="*" element={<div className="text-on-surface p-12 text-center">404 - Not Found</div>} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}
