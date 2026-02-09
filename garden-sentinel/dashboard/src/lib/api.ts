import axios from 'axios';
import { useAuthStore } from '../stores/authStore';

// Create axios instance
export const api = axios.create({
  baseURL: import.meta.env.VITE_API_URL || '',
  headers: {
    'Content-Type': 'application/json',
  },
});

// Add auth token to requests
api.interceptors.request.use((config) => {
  const token = useAuthStore.getState().token;
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// Handle auth errors
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      useAuthStore.getState().logout();
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// API functions
export const authApi = {
  login: (username: string, password: string) =>
    api.post('/api/auth/login', { username, password }),

  logout: () => api.post('/api/auth/logout'),

  me: () => api.get('/api/auth/me'),
};

export const mapApi = {
  getState: () => api.get('/api/map/state'),
  getZones: () => api.get('/api/map/zones'),
  getCameras: () => api.get('/api/map/cameras'),
  getTracks: () => api.get('/api/map/tracks'),
};

export const camerasApi = {
  list: () => api.get('/api/cameras'),
  get: (id: string) => api.get(`/api/cameras/${id}`),
  getStream: (id: string) => `/api/cameras/${id}/stream`,
  spray: (id: string) => api.post(`/api/cameras/${id}/spray`),
};

export const alertsApi = {
  list: (params?: { limit?: number; offset?: number }) =>
    api.get('/api/alerts', { params }),
  get: (id: string) => api.get(`/api/alerts/${id}`),
  acknowledge: (id: string) => api.post(`/api/alerts/${id}/acknowledge`),
};

export const analyticsApi = {
  getEffectiveness: () => api.get('/api/analytics/effectiveness'),
  getPatterns: (predatorType?: string) =>
    api.get('/api/analytics/patterns', { params: { predator_type: predatorType } }),
  getPredictions: () => api.get('/api/analytics/predictions'),
  detections: (params: { range: string }) =>
    api.get('/api/analytics/detections', { params }),
  deterrence: (params: { range: string }) =>
    api.get('/api/analytics/deterrence', { params }),
  predators: (params: { range: string }) =>
    api.get('/api/analytics/predators', { params }),
};

export const healthApi = {
  getDevices: () => api.get('/api/health/devices'),
  getDevice: (id: string) => api.get(`/api/health/devices/${id}`),
  getFleetOverview: () => api.get('/api/health/fleet'),
};

export const weatherApi = {
  getCurrent: () => api.get('/api/weather/current'),
  getForecast: () => api.get('/api/weather/forecast'),
  getActivityPrediction: (predatorType?: string) =>
    api.get('/api/weather/activity', { params: { predator_type: predatorType } }),
};

export const settingsApi = {
  get: () => api.get('/api/settings'),
  update: (data: any) => api.put('/api/settings', data),
};

export const usersApi = {
  list: () => api.get('/api/auth/users'),
  create: (data: { username: string; password: string; role: string }) =>
    api.post('/api/auth/users', data),
  delete: (userId: string) => api.delete(`/api/auth/users/${userId}`),
};

export const apiKeysApi = {
  list: () => api.get('/api/auth/api-keys'),
  create: (data: { name: string; device_id: string }) =>
    api.post('/api/auth/api-keys', data),
  delete: (keyId: string) => api.delete(`/api/auth/api-keys/${keyId}`),
};
