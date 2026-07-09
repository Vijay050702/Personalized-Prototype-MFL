import React from 'react';
import { NavLink } from 'react-router-dom';
import {
  LayoutDashboard, 
  Users, 
  Database, 
  Cpu, 
  Settings, 
  Bell, 
  Search,
  ChevronRight,
  Zap,
  Layers,
  ArrowRightLeft,
  Activity,
  BarChart3 as BarChartIcon,
  FlaskConical,
} from 'lucide-react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

const navItems = [
  { icon: LayoutDashboard, label: 'Dashboard', path: '/' },
  { icon: Users, label: 'Clients', path: '/clients' },
  { icon: Database, label: 'Datasets', path: '/datasets' },
  { icon: Cpu, label: 'Training', path: '/training' },
  { icon: Layers, label: 'Prototypes', path: '/prototypes' },
  { icon: ArrowRightLeft, label: 'Knowledge Transfer', path: '/knowledge-transfer' },
  { icon: Activity, label: 'Similarity Analysis', path: '/similarity' },
  { icon: FlaskConical, label: 'Experiments', path: '/experiments' },
  { icon: BarChartIcon, label: 'Model Evaluation', path: '/evaluation' },
];

export const Sidebar = () => {
  return (
    <aside className="w-[260px] h-screen bg-surface-container-low border-r border-outline-variant flex flex-col fixed left-0 top-0 z-50">
      <div className="p-6 flex items-center gap-3">
        <div className="w-10 h-10 bg-primary-container rounded-xl flex items-center justify-center text-primary shadow-lg shadow-primary/20">
          <Zap size={24} fill="currentColor" />
        </div>
        <div>
          <h1 className="font-display font-bold text-lg leading-tight tracking-tight text-on-surface">
            Federated
          </h1>
          <p className="text-[10px] uppercase tracking-[0.2em] font-mono text-outline font-semibold">
            Intelligence Sys
          </p>
        </div>
      </div>

      <nav className="flex-1 px-3 mt-4 space-y-1">
        {navItems.map((item) => (
          <NavLink
            key={item.path}
            to={item.path}
            className={({ isActive }) => cn(
              "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 group relative",
              isActive 
                ? "bg-primary/10 text-primary font-medium" 
                : "text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface"
            )}
          >
            {({ isActive }) => (
              <>
                <item.icon size={20} className={cn("transition-colors", isActive ? "text-primary" : "text-outline")} />
                <span className="text-sm">{item.label}</span>
                {isActive && (
                  <div className="absolute left-0 w-1 h-6 bg-primary rounded-r-full" />
                )}
                <ChevronRight 
                  size={14} 
                  className={cn(
                    "ml-auto opacity-0 -translate-x-2 transition-all group-hover:opacity-40 group-hover:translate-x-0",
                    isActive && "hidden"
                  )} 
                />
              </>
            )}
          </NavLink>
        ))}
      </nav>

      <div className="p-4 border-t border-outline-variant">
        <NavLink
          to="/settings"
          className={({ isActive }) => cn(
            "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-all duration-200 text-on-surface-variant hover:bg-surface-container-high hover:text-on-surface",
            isActive && "bg-primary/10 text-primary font-medium"
          )}
        >
          <Settings size={20} className="text-outline" />
          <span className="text-sm">Settings</span>
        </NavLink>
        
        <div className="mt-4 p-3 bg-surface-container rounded-xl border border-outline-variant/50">
          <div className="flex items-center gap-3 mb-2">
            <div className="w-8 h-8 rounded-full bg-gradient-to-tr from-secondary to-primary" />
            <div className="flex-1 min-w-0">
              <p className="text-xs font-medium truncate text-on-surface">Admin Console</p>
              <p className="text-[10px] text-outline truncate">v2.1.0-stable</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};
