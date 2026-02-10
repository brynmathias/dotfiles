import { useQuery } from '@tanstack/react-query';
import { healthApi, analyticsApi, alertsApi, weatherApi } from '../lib/api';
import {
  Activity,
  Camera,
  AlertTriangle,
  Thermometer,
  Droplets,
  Wind,
  Battery,
  Wifi,
  CheckCircle,
  XCircle,
} from 'lucide-react';
import { formatDistanceToNow } from 'date-fns';

interface StatCardProps {
  title: string;
  value: string | number;
  icon: React.ReactNode;
  change?: string;
  status?: 'good' | 'warning' | 'error';
}

function StatCard({ title, value, icon, change, status }: StatCardProps) {
  const statusColors = {
    good: 'text-green-500',
    warning: 'text-yellow-500',
    error: 'text-red-500',
  };

  return (
    <div className="bg-white rounded-lg shadow p-6">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-500">{title}</p>
          <p className={`text-2xl font-bold ${status ? statusColors[status] : ''}`}>
            {value}
          </p>
          {change && <p className="text-xs text-gray-400 mt-1">{change}</p>}
        </div>
        <div className={`p-3 rounded-full bg-gray-100 ${status ? statusColors[status] : 'text-gray-600'}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

function DeviceStatus({ device }: { device: any }) {
  const isOnline = device.status === 'online';

  return (
    <div className="flex items-center justify-between py-3 border-b last:border-0">
      <div className="flex items-center gap-3">
        <div className={`w-2 h-2 rounded-full ${isOnline ? 'bg-green-500' : 'bg-red-500'}`} />
        <div>
          <p className="font-medium">{device.name}</p>
          <p className="text-sm text-gray-500">{device.device_id}</p>
        </div>
      </div>
      <div className="text-right">
        <div className="flex items-center gap-2 text-sm">
          <Battery className="w-4 h-4" />
          <span>{device.battery_percent?.toFixed(0) || '--'}%</span>
        </div>
        <div className="flex items-center gap-2 text-sm text-gray-500">
          <Wifi className="w-4 h-4" />
          <span>{device.wifi_signal || '--'} dBm</span>
        </div>
      </div>
    </div>
  );
}

function RecentAlert({ alert }: { alert: any }) {
  const severityColors = {
    low: 'bg-blue-100 text-blue-800',
    medium: 'bg-yellow-100 text-yellow-800',
    high: 'bg-orange-100 text-orange-800',
    critical: 'bg-red-100 text-red-800',
  };

  return (
    <div className="flex items-center gap-3 py-3 border-b last:border-0">
      <div className={`px-2 py-1 rounded text-xs font-medium ${severityColors[alert.severity as keyof typeof severityColors] || 'bg-gray-100'}`}>
        {alert.severity}
      </div>
      <div className="flex-1">
        <p className="font-medium">{alert.predator_type || 'Detection'}</p>
        <p className="text-sm text-gray-500">{alert.camera_name}</p>
      </div>
      <p className="text-sm text-gray-400">
        {formatDistanceToNow(new Date(alert.timestamp * 1000), { addSuffix: true })}
      </p>
    </div>
  );
}

export default function Dashboard() {
  const { data: fleet } = useQuery({
    queryKey: ['fleetHealth'],
    queryFn: () => healthApi.getFleetOverview().then((res) => res.data),
    refetchInterval: 30000,
  });

  const { data: devices } = useQuery({
    queryKey: ['devices'],
    queryFn: () => healthApi.getDevices().then((res) => res.data),
    refetchInterval: 30000,
  });

  const { data: effectiveness } = useQuery({
    queryKey: ['effectiveness'],
    queryFn: () => analyticsApi.getEffectiveness().then((res) => res.data),
  });

  const { data: alerts } = useQuery({
    queryKey: ['recentAlerts'],
    queryFn: () => alertsApi.list({ limit: 5 }).then((res) => res.data),
    refetchInterval: 10000,
  });

  const { data: weather } = useQuery({
    queryKey: ['weather'],
    queryFn: () => weatherApi.getCurrent().then((res) => res.data),
    refetchInterval: 300000, // 5 minutes
  });

  const { data: activityPrediction } = useQuery({
    queryKey: ['activityPrediction'],
    queryFn: () => weatherApi.getActivityPrediction().then((res) => res.data),
    refetchInterval: 300000,
  });

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-2xl font-bold">Dashboard</h1>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <StatCard
          title="Devices Online"
          value={`${fleet?.online_count || 0}/${fleet?.total_count || 0}`}
          icon={<Camera className="w-6 h-6" />}
          status={fleet?.online_count === fleet?.total_count ? 'good' : 'warning'}
        />

        <StatCard
          title="Detections Today"
          value={fleet?.detections_today || 0}
          icon={<Activity className="w-6 h-6" />}
        />

        <StatCard
          title="Deterrence Rate"
          value={`${((effectiveness?.effectiveness || 0) * 100).toFixed(0)}%`}
          icon={effectiveness?.effectiveness >= 0.8 ? <CheckCircle className="w-6 h-6" /> : <AlertTriangle className="w-6 h-6" />}
          status={effectiveness?.effectiveness >= 0.8 ? 'good' : effectiveness?.effectiveness >= 0.5 ? 'warning' : 'error'}
        />

        <StatCard
          title="Active Alerts"
          value={alerts?.unacknowledged_count || 0}
          icon={<AlertTriangle className="w-6 h-6" />}
          status={alerts?.unacknowledged_count > 0 ? 'warning' : 'good'}
        />
      </div>

      {/* Two column layout */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Device Status */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="font-semibold">Device Status</h2>
          </div>
          <div className="p-4">
            {devices?.devices?.length > 0 ? (
              devices.devices.map((device: any) => (
                <DeviceStatus key={device.device_id} device={device} />
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No devices registered</p>
            )}
          </div>
        </div>

        {/* Recent Alerts */}
        <div className="bg-white rounded-lg shadow">
          <div className="p-4 border-b">
            <h2 className="font-semibold">Recent Alerts</h2>
          </div>
          <div className="p-4">
            {alerts?.alerts?.length > 0 ? (
              alerts.alerts.map((alert: any) => (
                <RecentAlert key={alert.id} alert={alert} />
              ))
            ) : (
              <p className="text-gray-500 text-center py-4">No recent alerts</p>
            )}
          </div>
        </div>
      </div>

      {/* Weather and Activity */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Current Weather */}
        <div className="bg-white rounded-lg shadow p-4">
          <h2 className="font-semibold mb-4">Current Weather</h2>
          {weather ? (
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Thermometer className="w-5 h-5 text-gray-500" />
                  <span>Temperature</span>
                </div>
                <span className="font-medium">{weather.temperature_c?.toFixed(1)}°C</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Droplets className="w-5 h-5 text-gray-500" />
                  <span>Humidity</span>
                </div>
                <span className="font-medium">{weather.humidity?.toFixed(0)}%</span>
              </div>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2">
                  <Wind className="w-5 h-5 text-gray-500" />
                  <span>Wind</span>
                </div>
                <span className="font-medium">{weather.wind_speed_ms?.toFixed(1)} m/s</span>
              </div>
              <div className="pt-2 border-t">
                <p className="text-sm text-gray-600 capitalize">{weather.description}</p>
              </div>
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">Weather data unavailable</p>
          )}
        </div>

        {/* Activity Prediction */}
        <div className="bg-white rounded-lg shadow p-4 lg:col-span-2">
          <h2 className="font-semibold mb-4">Predator Activity Prediction</h2>
          {activityPrediction ? (
            <div>
              <div className="flex items-center gap-4 mb-4">
                <div className="flex-1">
                  <div className="flex items-center justify-between mb-1">
                    <span className="text-sm">Activity Level</span>
                    <span className="text-sm font-medium">
                      {(activityPrediction.activity_level * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="w-full bg-gray-200 rounded-full h-2">
                    <div
                      className={`h-2 rounded-full ${
                        activityPrediction.activity_level > 0.7
                          ? 'bg-red-500'
                          : activityPrediction.activity_level > 0.4
                          ? 'bg-yellow-500'
                          : 'bg-green-500'
                      }`}
                      style={{ width: `${activityPrediction.activity_level * 100}%` }}
                    />
                  </div>
                </div>
                <div className="text-sm text-gray-500 capitalize">
                  {activityPrediction.time_of_day}
                </div>
              </div>

              {activityPrediction.risk_factors?.length > 0 && (
                <div className="mb-3">
                  <p className="text-sm font-medium text-gray-700 mb-1">Risk Factors:</p>
                  <ul className="text-sm text-gray-600 list-disc list-inside">
                    {activityPrediction.risk_factors.map((factor: string, i: number) => (
                      <li key={i}>{factor}</li>
                    ))}
                  </ul>
                </div>
              )}

              {activityPrediction.recommendations?.length > 0 && (
                <div>
                  <p className="text-sm font-medium text-gray-700 mb-1">Recommendations:</p>
                  <ul className="text-sm text-gray-600 list-disc list-inside">
                    {activityPrediction.recommendations.map((rec: string, i: number) => (
                      <li key={i}>{rec}</li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          ) : (
            <p className="text-gray-500 text-center py-4">Prediction unavailable</p>
          )}
        </div>
      </div>
    </div>
  );
}
