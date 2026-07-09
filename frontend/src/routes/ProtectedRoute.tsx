import { type ReactNode } from 'react';
import { Navigate, useLocation } from 'react-router-dom';
import { useAuth } from '../hooks/useAuth';
import { AuthBanner } from '../components/auth/AuthBanner';

interface ProtectedRouteProps {
  children: ReactNode;
  requireAuth?: boolean;
}

export const ProtectedRoute = ({ children, requireAuth = false }: ProtectedRouteProps) => {
  const { mode, isAuthenticated, isLoading } = useAuth();
  const location = useLocation();

  if (isLoading) {
    return (
      <div className="flex items-center justify-center min-h-[60vh]">
        <div className="flex flex-col items-center gap-4">
          <div className="w-8 h-8 border-2 border-primary border-t-transparent rounded-full animate-spin" />
          <p className="text-sm text-outline font-mono">Verifying session...</p>
        </div>
      </div>
    );
  }

  if (mode === 'disabled') {
    return (
      <>
        <AuthBanner />
        {children}
      </>
    );
  }

  if (mode === 'enabled' && !isAuthenticated) {
    if (requireAuth) {
      return <Navigate to="/login" state={{ from: location }} replace />;
    }
    return (
      <>
        <AuthBanner />
        {children}
      </>
    );
  }

  return <>{children}</>;
};
