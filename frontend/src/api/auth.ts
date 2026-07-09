import type {
  AuthStatusResponse,
  AuthUser,
  LoginRequest,
  LoginResponse,
  RefreshResponse,
  StorageType,
} from '../types';
import client from './axios';

export interface TokenStorage {
  getAccessToken(): string | null;
  setAccessToken(token: string | null): void;
  getRefreshToken(): string | null;
  setRefreshToken(token: string | null): void;
  getStoredUser(): AuthUser | null;
  setStoredUser(user: AuthUser | null): void;
  clear(): void;
}

function createLocalStorageStorage(): TokenStorage {
  const ACCESS_KEY = 'auth_access_token';
  const REFRESH_KEY = 'auth_refresh_token';
  const USER_KEY = 'auth_user';
  return {
    getAccessToken(): string | null { try { return localStorage.getItem(ACCESS_KEY); } catch { return null; } },
    setAccessToken(token: string | null): void { try { if (token) localStorage.setItem(ACCESS_KEY, token); else localStorage.removeItem(ACCESS_KEY); } catch { /* noop */ } },
    getRefreshToken(): string | null { try { return localStorage.getItem(REFRESH_KEY); } catch { return null; } },
    setRefreshToken(token: string | null): void { try { if (token) localStorage.setItem(REFRESH_KEY, token); else localStorage.removeItem(REFRESH_KEY); } catch { /* noop */ } },
    getStoredUser(): AuthUser | null { try { const raw = localStorage.getItem(USER_KEY); return raw ? JSON.parse(raw) as AuthUser : null; } catch { return null; } },
    setStoredUser(user: AuthUser | null): void { try { if (user) localStorage.setItem(USER_KEY, JSON.stringify(user)); else localStorage.removeItem(USER_KEY); } catch { /* noop */ } },
    clear(): void { try { localStorage.removeItem(ACCESS_KEY); localStorage.removeItem(REFRESH_KEY); localStorage.removeItem(USER_KEY); } catch { /* noop */ } },
  };
}

function createSessionStorageStorage(): TokenStorage {
  const ACCESS_KEY = 'auth_access_token';
  const REFRESH_KEY = 'auth_refresh_token';
  const USER_KEY = 'auth_user';
  return {
    getAccessToken(): string | null { try { return sessionStorage.getItem(ACCESS_KEY); } catch { return null; } },
    setAccessToken(token: string | null): void { try { if (token) sessionStorage.setItem(ACCESS_KEY, token); else sessionStorage.removeItem(ACCESS_KEY); } catch { /* noop */ } },
    getRefreshToken(): string | null { try { return sessionStorage.getItem(REFRESH_KEY); } catch { return null; } },
    setRefreshToken(token: string | null): void { try { if (token) sessionStorage.setItem(REFRESH_KEY, token); else sessionStorage.removeItem(REFRESH_KEY); } catch { /* noop */ } },
    getStoredUser(): AuthUser | null { try { const raw = sessionStorage.getItem(USER_KEY); return raw ? JSON.parse(raw) as AuthUser : null; } catch { return null; } },
    setStoredUser(user: AuthUser | null): void { try { if (user) sessionStorage.setItem(USER_KEY, JSON.stringify(user)); else sessionStorage.removeItem(USER_KEY); } catch { /* noop */ } },
    clear(): void { try { sessionStorage.removeItem(ACCESS_KEY); sessionStorage.removeItem(REFRESH_KEY); sessionStorage.removeItem(USER_KEY); } catch { /* noop */ } },
  };
}

function createMemoryStorage(): TokenStorage {
  let accessToken: string | null = null;
  let refreshToken: string | null = null;
  let storedUser: AuthUser | null = null;
  return {
    getAccessToken(): string | null { return accessToken; },
    setAccessToken(token: string | null): void { accessToken = token; },
    getRefreshToken(): string | null { return refreshToken; },
    setRefreshToken(token: string | null): void { refreshToken = token; },
    getStoredUser(): AuthUser | null { return storedUser; },
    setStoredUser(user: AuthUser | null): void { storedUser = user; },
    clear(): void { accessToken = null; refreshToken = null; storedUser = null; },
  };
}

export function createTokenStorage(type: StorageType = 'local'): TokenStorage {
  switch (type) {
    case 'session': return createSessionStorageStorage();
    case 'memory': return createMemoryStorage();
    case 'local': return createLocalStorageStorage();
  }
}

export const defaultTokenStorage: TokenStorage = (() => {
  try {
    localStorage.getItem('__probe__');
    return createLocalStorageStorage();
  } catch {
    return createMemoryStorage();
  }
})();

const AUTH_BASE = '/api/v1/auth';

export async function probeAuthEndpoints(): Promise<boolean> {
  try {
    await client.get<AuthStatusResponse>(`${AUTH_BASE}/status`, { timeout: 5000 });
    return true;
  } catch {
    return false;
  }
}

export async function login(credentials: LoginRequest): Promise<LoginResponse> {
  const response = await client.post<LoginResponse>(`${AUTH_BASE}/login`, credentials);
  return response.data;
}

export async function logout(): Promise<void> {
  try {
    await client.post(`${AUTH_BASE}/logout`);
  } catch {
    // Silently ignore — session is cleared locally regardless
  }
}

export async function refreshToken(token: string): Promise<RefreshResponse> {
  const response = await client.post<RefreshResponse>(`${AUTH_BASE}/refresh`, { refresh_token: token });
  return response.data;
}

export async function fetchCurrentUser(): Promise<AuthUser> {
  const response = await client.get<AuthUser>(`${AUTH_BASE}/me`);
  return response.data;
}

export function clearAuthData(): void {
  defaultTokenStorage.clear();
}
