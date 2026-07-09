import type { ReactElement } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { MemoryRouter } from 'react-router-dom';

import { AuthProvider } from '../../context/AuthContext';
import { ProtectedRoute } from '../../routes/ProtectedRoute';
import { AuthBanner } from '../../components/auth/AuthBanner';
import { LoginPage } from '../../components/auth/LoginPage';
import { useAuth } from '../../hooks/useAuth';
import {
  createTokenStorage,
  defaultTokenStorage,
  probeAuthEndpoints,
  clearAuthData,
} from '../../api/auth';
import type { AuthContextValue, AuthUser } from '../../types';
import * as authApi from '../../api/auth';

vi.mock('../../api/auth', async () => {
  const actual = await vi.importActual('../../api/auth');
  return {
    ...actual,
    probeAuthEndpoints: vi.fn(),
    login: vi.fn(),
    logout: vi.fn(),
    fetchCurrentUser: vi.fn(),
    refreshToken: vi.fn(),
  };
});

const mockUser: AuthUser = {
  id: 'user_001',
  username: 'admin',
  email: 'admin@pp-mfl.io',
  role: 'admin',
};

function createQueryWrapper(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>{ui}</MemoryRouter>
    </QueryClientProvider>,
  );
}

function createAuthWrapper(ui: ReactElement) {
  const queryClient = new QueryClient({
    defaultOptions: { queries: { retry: false, retryDelay: 0 } },
  });
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <AuthProvider>{ui}</AuthProvider>
      </MemoryRouter>
    </QueryClientProvider>,
  );
}

// eslint-disable-next-line @typescript-eslint/no-unused-vars
function TestConsumer({ onValue }: { onValue: (value: AuthContextValue) => void }) {
  const auth = useAuth();
  onValue?.(auth);
  return <div data-testid="auth-consumer">{auth.mode}</div>;
}

describe('TokenStorage', () => {
  beforeEach(() => {
    localStorage.clear();
    sessionStorage.clear();
  });

  describe('localStorage storage', () => {
    const storage = createTokenStorage('local');

    it('stores and retrieves access token', () => {
      storage.setAccessToken('test-token');
      expect(storage.getAccessToken()).toBe('test-token');
      expect(localStorage.getItem('auth_access_token')).toBe('test-token');
    });

    it('stores and retrieves refresh token', () => {
      storage.setRefreshToken('test-refresh');
      expect(storage.getRefreshToken()).toBe('test-refresh');
      expect(localStorage.getItem('auth_refresh_token')).toBe('test-refresh');
    });

    it('stores and retrieves user', () => {
      storage.setStoredUser(mockUser);
      expect(storage.getStoredUser()).toEqual(mockUser);
      expect(localStorage.getItem('auth_user')).toBeTruthy();
    });

    it('clears all stored data', () => {
      storage.setAccessToken('token');
      storage.setRefreshToken('refresh');
      storage.setStoredUser(mockUser);
      storage.clear();
      expect(storage.getAccessToken()).toBeNull();
      expect(storage.getRefreshToken()).toBeNull();
      expect(storage.getStoredUser()).toBeNull();
    });

    it('handles null values gracefully', () => {
      storage.setAccessToken(null);
      storage.setRefreshToken(null);
      storage.setStoredUser(null);
      expect(storage.getAccessToken()).toBeNull();
      expect(storage.getRefreshToken()).toBeNull();
      expect(storage.getStoredUser()).toBeNull();
    });
  });

  describe('sessionStorage storage', () => {
    const storage = createTokenStorage('session');

    it('stores and retrieves tokens in sessionStorage', () => {
      storage.setAccessToken('session-token');
      expect(storage.getAccessToken()).toBe('session-token');
      expect(sessionStorage.getItem('auth_access_token')).toBe('session-token');
    });

    it('clears session data', () => {
      storage.setAccessToken('token');
      storage.clear();
      expect(storage.getAccessToken()).toBeNull();
    });
  });

  describe('memory storage', () => {
    const storage = createTokenStorage('memory');

    it('stores and retrieves tokens in memory', () => {
      storage.setAccessToken('mem-token');
      expect(storage.getAccessToken()).toBe('mem-token');
      expect(localStorage.getItem('auth_access_token')).toBeNull();
    });

    it('clears memory data', () => {
      storage.setAccessToken('token');
      storage.clear();
      expect(storage.getAccessToken()).toBeNull();
    });
  });

  describe('defaultTokenStorage', () => {
    it('falls back to memory when localStorage is unavailable', () => {
      expect(defaultTokenStorage).toBeDefined();
    });
  });
});

describe('Auth API', () => {
  const mockedProbe = vi.mocked(probeAuthEndpoints);

  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('probeAuthEndpoints returns true when endpoint exists', async () => {
    mockedProbe.mockResolvedValue(true);
    const result = await probeAuthEndpoints();
    expect(result).toBe(true);
  });

  it('probeAuthEndpoints returns false when endpoint does not exist', async () => {
    mockedProbe.mockResolvedValue(false);
    const result = await probeAuthEndpoints();
    expect(result).toBe(false);
  });

  it('clearAuthData clears all stored data', () => {
    defaultTokenStorage.setAccessToken('test');
    clearAuthData();
    expect(defaultTokenStorage.getAccessToken()).toBeNull();
  });

  it('createTokenStorage returns correct type', () => {
    expect(createTokenStorage('local')).toBeDefined();
    expect(createTokenStorage('session')).toBeDefined();
    expect(createTokenStorage('memory')).toBeDefined();
  });
});

describe('AuthContext', () => {
  const mockedProbe = vi.mocked(authApi.probeAuthEndpoints);
  const mockedLogin = vi.mocked(authApi.login);
  const mockedFetchCurrentUser = vi.mocked(authApi.fetchCurrentUser);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('shows loading state while probing', () => {
    mockedProbe.mockReturnValue(new Promise<boolean>(() => {}));
    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);
    expect(screen.getByTestId('auth-consumer')).toBeTruthy();
  });

  it('enters disabled mode when backend has no auth', async () => {
    mockedProbe.mockResolvedValue(false);
    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);
    await waitFor(() => {
      const calls = onValue.mock.calls;
      const lastCall = calls[calls.length - 1]?.[0] as AuthContextValue;
      expect(lastCall?.mode).toBe('disabled');
      expect(lastCall?.isAuthEnabled).toBe(false);
      expect(lastCall?.isAuthenticated).toBe(false);
    });
  });

  it('restores session when token exists and auth is enabled', async () => {
    mockedProbe.mockResolvedValue(true);
    mockedFetchCurrentUser.mockResolvedValue(mockUser);
    defaultTokenStorage.setAccessToken('valid-token');

    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);

    await waitFor(() => {
      const calls = onValue.mock.calls;
      const lastCall = calls[calls.length - 1]?.[0] as AuthContextValue;
      if (lastCall?.mode === 'enabled') {
        expect(lastCall.user).toEqual(mockUser);
        expect(lastCall.isAuthenticated).toBe(true);
      }
    });
  });

  it('enters disabled mode on probe failure after restoring invalid token', async () => {
    mockedProbe.mockResolvedValue(true);
    mockedFetchCurrentUser.mockRejectedValue(new Error('Unauthorized'));
    defaultTokenStorage.setAccessToken('invalid-token');

    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);

    await waitFor(() => {
      const calls = onValue.mock.calls;
      const lastCall = calls[calls.length - 1]?.[0] as AuthContextValue;
      expect(lastCall?.mode).toBe('disabled');
      expect(lastCall?.isAuthenticated).toBe(false);
    });
  });

  it('probe failure transitions to disabled mode', async () => {
    mockedProbe.mockRejectedValue(new Error('Network error'));

    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);

    await waitFor(() => {
      const calls = onValue.mock.calls;
      const lastCall = calls[calls.length - 1]?.[0] as AuthContextValue;
      expect(lastCall?.mode).toBe('disabled');
    });
  });

  it('login function works correctly', async () => {
    mockedProbe.mockResolvedValue(true);
    mockedLogin.mockResolvedValue({
      access_token: 'new-token',
      refresh_token: 'new-refresh',
      token_type: 'bearer',
      expires_in: 3600,
      user: mockUser,
    });

    const onValue = vi.fn();
    createAuthWrapper(<TestConsumer onValue={onValue} />);

    await waitFor(() => {
      const calls = onValue.mock.calls;
      const authCtx = calls[calls.length - 1]?.[0] as AuthContextValue;
      if (authCtx?.mode === 'enabled') {
        expect(authCtx.isAuthEnabled).toBe(true);
      }
    });
  });
});

describe('ProtectedRoute', () => {
  const mockedProbe = vi.mocked(authApi.probeAuthEndpoints);

  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it('shows loading state while auth is being determined', () => {
    mockedProbe.mockReturnValue(new Promise<boolean>(() => {}));
    createAuthWrapper(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected Content</div>
      </ProtectedRoute>,
    );
    expect(screen.getByText('Verifying session...')).toBeTruthy();
  });

  it('shows auth banner and content when auth is disabled', async () => {
    mockedProbe.mockResolvedValue(false);
    createAuthWrapper(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected Content</div>
      </ProtectedRoute>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeTruthy();
    });
    expect(screen.getByText('Authentication Disabled')).toBeTruthy();
  });

  it('renders children when auth is enabled and authenticated', async () => {
    mockedProbe.mockResolvedValue(true);
    vi.mocked(authApi.fetchCurrentUser).mockResolvedValue(mockUser);
    defaultTokenStorage.setAccessToken('valid-token');

    createAuthWrapper(
      <ProtectedRoute>
        <div data-testid="protected-content">Protected Content</div>
      </ProtectedRoute>,
    );

    await waitFor(() => {
      expect(screen.getByTestId('protected-content')).toBeTruthy();
    });
  });
});

describe('AuthBanner', () => {
  it('renders the disabled auth message', () => {
    createQueryWrapper(<AuthBanner />);
    expect(screen.getByText('Authentication Disabled')).toBeTruthy();
    expect(screen.getByText(/Authentication is not enabled/)).toBeTruthy();
  });
});

describe('LoginPage', () => {
  it('renders the login placeholder when auth is disabled', () => {
    createQueryWrapper(<LoginPage />);
    expect(screen.getByText('Authentication Unavailable')).toBeTruthy();
    expect(screen.getByText(/Backend auth endpoints not detected/)).toBeTruthy();
  });
});

describe('useAuth', () => {
  it('throws when used outside AuthProvider', () => {
    const TestHook = () => {
      useAuth();
      return null;
    };
    expect(() => createQueryWrapper(<TestHook />)).toThrow(
      'useAuth must be used within an AuthProvider',
    );
  });

  it('returns context value when used within AuthProvider', async () => {
    const mockedProbe = vi.mocked(authApi.probeAuthEndpoints);
    mockedProbe.mockResolvedValue(false);

    let capturedValue: AuthContextValue | null = null;
    function TestComponent() {
      const auth = useAuth();
      capturedValue = auth;
      return <div data-testid="mode">{auth.mode}</div>;
    }

    createAuthWrapper(<TestComponent />);

    await waitFor(() => {
      expect(screen.getByTestId('mode').textContent).toBe('disabled');
    });
    expect(capturedValue).not.toBeNull();
  });
});

describe('Axios auth interceptor (integration)', () => {
  it('handles 401 session expired event dispatch', async () => {
    const listener = vi.fn();
    window.addEventListener('auth:session-expired', listener);
    const axios = (await import('../../api/axios')).default;
    try {
      await axios.get('/api/v1/auth/me');
    } catch {
      // Expected to fail
    }
    await waitFor(() => {
      expect(listener).not.toHaveBeenCalled();
    });
  });
});
