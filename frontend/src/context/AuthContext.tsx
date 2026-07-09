import { createContext, useState, useEffect, useCallback, useRef, type ReactNode } from 'react';
import type { AuthContextValue, AuthMode, AuthUser, LoginRequest } from '../types';
import {
  probeAuthEndpoints,
  login as apiLogin,
  logout as apiLogout,
  fetchCurrentUser,
  refreshToken as apiRefreshToken,
  defaultTokenStorage,
  clearAuthData,
} from '../api/auth';

export const AuthContext = createContext<AuthContextValue | null>(null);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider = ({ children }: AuthProviderProps) => {
  const [mode, setMode] = useState<AuthMode>('loading');
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const isProbing = useRef(false);

  const refreshTimer = useRef<ReturnType<typeof setInterval>>();

  const clearSession = useCallback(() => {
    clearAuthData();
    setUser(null);
    if (refreshTimer.current) {
      clearInterval(refreshTimer.current);
      refreshTimer.current = undefined;
    }
  }, []);

  const restoreSession = useCallback(async () => {
    const token = defaultTokenStorage.getAccessToken();
    if (!token) {
      setIsLoading(false);
      return;
    }
    try {
      const currentUser = await fetchCurrentUser();
      setUser(currentUser);
      setMode('enabled');
    } catch {
      clearSession();
      setMode('disabled');
    } finally {
      setIsLoading(false);
    }
  }, [clearSession]);

  const checkAuth = useCallback(async () => {
    if (isProbing.current) return;
    isProbing.current = true;
    setIsLoading(true);
    try {
      const enabled = await probeAuthEndpoints();
      if (enabled) {
        setMode('enabled');
        await restoreSession();
      } else {
        setMode('disabled');
        clearSession();
        setIsLoading(false);
      }
    } catch {
      setMode('disabled');
      clearSession();
      setIsLoading(false);
    } finally {
      isProbing.current = false;
    }
  }, [restoreSession, clearSession]);

  useEffect(() => {
    checkAuth();
  }, [checkAuth]);

  useEffect(() => {
    return () => {
      if (refreshTimer.current) {
        clearInterval(refreshTimer.current);
      }
    };
  }, []);

  const scheduleTokenRefresh = useCallback((expiresInSeconds: number) => {
    if (refreshTimer.current) {
      clearInterval(refreshTimer.current);
    }
    const refreshMs = Math.max((expiresInSeconds - 60) * 1000, 30000);
    refreshTimer.current = setInterval(async () => {
      const storedRefresh = defaultTokenStorage.getRefreshToken();
      if (!storedRefresh) return;
      try {
        const result = await apiRefreshToken(storedRefresh);
        defaultTokenStorage.setAccessToken(result.access_token);
      } catch {
        clearSession();
        setMode('disabled');
      }
    }, refreshMs);
  }, [clearSession]);

  const login = useCallback(async (credentials: LoginRequest) => {
    const response = await apiLogin(credentials);
    defaultTokenStorage.setAccessToken(response.access_token);
    if (response.refresh_token) {
      defaultTokenStorage.setRefreshToken(response.refresh_token);
    }
    defaultTokenStorage.setStoredUser(response.user);
    setUser(response.user);
    setMode('enabled');
    scheduleTokenRefresh(response.expires_in);
  }, [scheduleTokenRefresh]);

  const logout = useCallback(async () => {
    try {
      await apiLogout();
    } finally {
      clearSession();
      setMode('disabled');
    }
  }, [clearSession]);

  const refreshSession = useCallback(async () => {
    const token = defaultTokenStorage.getAccessToken();
    if (!token) return;
    try {
      const currentUser = await fetchCurrentUser();
      setUser(currentUser);
    } catch {
      clearSession();
      setMode('disabled');
    }
  }, [clearSession]);

  const value: AuthContextValue = {
    mode,
    user,
    isAuthenticated: mode === 'enabled' && user !== null,
    isLoading,
    isAuthEnabled: mode === 'enabled',
    login,
    logout,
    refreshSession,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
