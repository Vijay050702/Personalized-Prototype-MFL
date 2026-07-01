import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface StatusBadgeProps {
  status: string;
  className?: string;
}

export const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className }) => {
  const getStyles = () => {
    const s = status.toLowerCase();
    if (s === 'active' || s === 'completed' || s === 'success') {
      return "bg-emerald-500/10 text-emerald-400 border-emerald-500/20";
    }
    if (s === 'inactive' || s === 'pending') {
      return "bg-amber-500/10 text-amber-400 border-amber-500/20";
    }
    if (s === 'error' || s === 'failed') {
      return "bg-rose-500/10 text-rose-400 border-rose-500/20";
    }
    if (s === 'running') {
      return "bg-primary-container/20 text-primary border-primary/30";
    }
    return "bg-surface-container-highest text-outline border-outline-variant";
  };

  return (
    <span className={cn(
      "px-2.5 py-1 rounded-full text-[10px] font-bold uppercase tracking-wider border",
      getStyles(),
      className
    )}>
      {status}
    </span>
  );
};
