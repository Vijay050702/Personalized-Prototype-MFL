import React from 'react';
import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

interface CardProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  variant?: 'default' | 'glass' | 'outline';
}

export const Card: React.FC<CardProps> = ({ 
  children, 
  className, 
  variant = 'default',
  ...props 
}) => {
  const variants = {
    default: "bg-surface-container border border-outline-variant/50 shadow-sm",
    glass: "glass-panel shadow-xl",
    outline: "bg-transparent border border-outline-variant",
  };

  return (
    <div 
      className={cn(
        "rounded-2xl overflow-hidden",
        variants[variant],
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
};
