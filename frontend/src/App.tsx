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
          <Route path="/settings" element={<div className="text-on-surface p-12 text-center border-2 border-dashed border-outline-variant rounded-3xl">System Settings - Configuration Module Coming Soon</div>} />
          <Route path="*" element={<div className="text-on-surface p-12 text-center">404 - Not Found</div>} />
        </Routes>
      </MainLayout>
    </BrowserRouter>
  );
}
