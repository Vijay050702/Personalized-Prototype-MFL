import React from 'react';
import { Sidebar } from './Sidebar';
import { Header } from './Header';
import { motion, AnimatePresence } from 'motion/react';
import { useLocation } from 'react-router-dom';

interface MainLayoutProps {
  children: React.ReactNode;
}

export const MainLayout: React.FC<MainLayoutProps> = ({ children }) => {
  const location = useLocation();

  return (
    <div className="flex min-h-screen bg-surface">
      <Sidebar />
      <div className="flex-1 ml-[260px] flex flex-col min-h-screen relative">
        <Header />
        <main className="flex-1 p-8 overflow-y-auto">
          <AnimatePresence mode="wait">
            <motion.div
              key={location.pathname}
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -10 }}
              transition={{ duration: 0.3, ease: [0.22, 1, 0.36, 1] }}
              className="max-w-(--spacing-max-content-width) mx-auto w-full"
            >
              {children}
            </motion.div>
          </AnimatePresence>
        </main>
        
        {/* Subtle Background Elements */}
        <div className="fixed top-0 right-0 w-1/2 h-1/2 bg-primary/5 blur-[120px] pointer-events-none -z-10" />
        <div className="fixed bottom-0 left-[260px] w-1/3 h-1/3 bg-secondary/5 blur-[100px] pointer-events-none -z-10" />
      </div>
    </div>
  );
};
