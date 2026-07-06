import axios, { AxiosError, type AxiosInstance, type InternalAxiosRequestConfig } from 'axios';

const BASE_URL: string = import.meta.env.VITE_API_URL ?? 'http://localhost:8000';
const TIMEOUT_MS: number = 15_000;

const client: AxiosInstance = axios.create({
  baseURL: BASE_URL,
  timeout: TIMEOUT_MS,
  headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
});

client.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token: string | null = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error: AxiosError) => Promise.reject(error),
);

client.interceptors.response.use(
  (response) => response,
  (error: AxiosError) => {
    if (error.code === 'ECONNABORTED') {
      return Promise.reject(new Error('Request timed out. Please try again.'));
    }
    if (!error.response) {
      return Promise.reject(
        new Error('Backend is unavailable. Please check your connection.'),
      );
    }
    const { status } = error.response;
    if (status === 404) {
      return Promise.reject(new Error('Resource not found.'));
    }
    if (status >= 500) {
      return Promise.reject(new Error('Server error. Please try again later.'));
    }
    return Promise.reject(error);
  },
);

export default client;
