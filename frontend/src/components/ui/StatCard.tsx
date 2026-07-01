import React from 'react';
import { TrendingUp, TrendingDown, Minus } from 'lucide-react';
import { Card } from './Card';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface StatCardProps {
  label: string;
  value: string | number;
  change?: number;
  trend?: 'up' | 'down' | 'neutral';
  icon?: React.ReactNode;
}

export const StatCard: React.FC<StatCardProps> = ({ label, value, change, trend, icon }) => {
  return (
    <Card className="p-5 flex flex-col gap-3">
      <div className="flex items-center justify-between">
        <span className="text-xs font-medium text-outline uppercase tracking-wider">{label}</span>
        {icon && <div className="text-primary">{icon}</div>}
      </div>
      <div className="flex items-end justify-between gap-2">
        <h3 className="text-2xl font-display font-bold text-on-surface tracking-tight leading-none">{value}</h3>
        {change !== undefined && (
          <div className={cn(
            "flex items-center gap-1 text-xs font-medium px-2 py-1 rounded-full",
            trend === 'up' ? "text-emerald-400 bg-emerald-400/10" : 
            trend === 'down' ? "text-rose-400 bg-rose-400/10" : 
            "text-outline bg-surface-container-high"
          )}>
            {trend === 'up' ? <TrendingUp size={12} /> : 
             trend === 'down' ? <TrendingDown size={12} /> : 
             <Minus size={12} />}
            <span>{trend === 'neutral' ? 'Stable' : `${change}%`}</span>
          </div>
        )}
      </div>
    </Card>
  );
};
