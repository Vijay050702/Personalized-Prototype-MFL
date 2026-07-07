import React from 'react';
import { useLocation } from 'react-router-dom';
import { Search, Bell, HelpCircle, User } from 'lucide-react';

export const Header = () => {
  const location = useLocation();
  
  const getPageTitle = () => {
    const path = location.pathname;
    if (path === '/') return 'System Overview';
    if (path === '/clients') return 'Client Network';
    if (path === '/datasets') return 'Data Repository';
    if (path === '/training') return 'Training Intelligence';
    if (path === '/prototypes') return 'Prototype Repository';
    if (path === '/knowledge-transfer') return 'Knowledge Transfer';
    if (path === '/settings') return 'System Configuration';
    return 'Dashboard';
  };

  return (
    <header className="h-16 border-b border-outline-variant bg-surface/80 backdrop-blur-md flex items-center justify-between px-8 sticky top-0 z-40">
      <div>
        <h2 className="font-display text-lg font-semibold text-on-surface tracking-tight">
          {getPageTitle()}
        </h2>
        <div className="flex items-center gap-2 mt-0.5">
          <div className="w-1.5 h-1.5 rounded-full bg-secondary animate-pulse" />
          <span className="text-[10px] uppercase tracking-wider text-outline font-mono">System Live • Nodes: 1,284</span>
        </div>
      </div>

      <div className="flex items-center gap-4">
        <div className="relative group hidden md:block">
          <Search size={18} className="absolute left-3 top-1/2 -translate-y-1/2 text-outline group-focus-within:text-primary transition-colors" />
          <input 
            type="text" 
            placeholder="Search system..." 
            className="w-64 pl-10 pr-4 py-1.5 bg-surface-container-lowest border border-outline-variant rounded-full text-sm focus:outline-none focus:ring-1 focus:ring-primary focus:border-primary transition-all"
          />
        </div>

        <div className="flex items-center gap-2 border-l border-outline-variant pl-4 ml-2">
          <button className="p-2 text-on-surface-variant hover:bg-surface-container-high rounded-full transition-colors relative">
            <Bell size={20} />
            <span className="absolute top-2 right-2 w-2 h-2 bg-primary rounded-full border-2 border-surface" />
          </button>
          <button className="p-2 text-on-surface-variant hover:bg-surface-container-high rounded-full transition-colors">
            <HelpCircle size={20} />
          </button>
          <div className="w-8 h-8 rounded-full bg-surface-container-highest flex items-center justify-center border border-outline-variant cursor-pointer hover:border-primary transition-colors ml-2">
            <User size={18} className="text-on-surface-variant" />
          </div>
        </div>
      </div>
    </header>
  );
};
