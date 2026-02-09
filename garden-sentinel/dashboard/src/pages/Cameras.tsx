import { useQuery } from '@tanstack/react-query';
import { camerasApi, healthApi } from '../lib/api';
import { Camera, Battery, Wifi, Play, Droplets } from 'lucide-react';
import { useState } from 'react';

export default function Cameras() {
  const [selectedCamera, setSelectedCamera] = useState<string | null>(null);

  const { data: devices } = useQuery({
    queryKey: ['devices'],
    queryFn: () => healthApi.getDevices().then((res) => res.data),
    refetchInterval: 10000,
  });

  const handleSpray = async (cameraId: string) => {
    if (confirm('Activate spray on this camera?')) {
      try {
        await camerasApi.spray(cameraId);
      } catch (err) {
        console.error('Failed to activate spray:', err);
      }
    }
  };

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Cameras</h1>

      <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-6">
        {devices?.devices?.map((device: any) => (
          <div key={device.device_id} className="bg-white rounded-lg shadow overflow-hidden">
            {/* Video stream placeholder */}
            <div className="aspect-video bg-gray-900 relative">
              {selectedCamera === device.device_id ? (
                <img
                  src={camerasApi.getStream(device.device_id)}
                  alt={device.name}
                  className="w-full h-full object-contain"
                />
              ) : (
                <div className="absolute inset-0 flex items-center justify-center">
                  <button
                    onClick={() => setSelectedCamera(device.device_id)}
                    className="bg-white/20 hover:bg-white/30 rounded-full p-4 transition"
                  >
                    <Play className="w-8 h-8 text-white" />
                  </button>
                </div>
              )}

              {/* Status indicator */}
              <div className="absolute top-2 right-2">
                <div
                  className={`w-3 h-3 rounded-full ${
                    device.status === 'online' ? 'bg-green-500' : 'bg-red-500'
                  }`}
                />
              </div>
            </div>

            {/* Camera info */}
            <div className="p-4">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold">{device.name || device.device_id}</h3>
                <span
                  className={`text-xs px-2 py-1 rounded ${
                    device.status === 'online'
                      ? 'bg-green-100 text-green-800'
                      : 'bg-red-100 text-red-800'
                  }`}
                >
                  {device.status}
                </span>
              </div>

              {/* Stats */}
              <div className="grid grid-cols-3 gap-2 text-sm text-gray-600 mb-4">
                <div className="flex items-center gap-1">
                  <Battery className="w-4 h-4" />
                  <span>{device.battery_percent?.toFixed(0) || '--'}%</span>
                </div>
                <div className="flex items-center gap-1">
                  <Wifi className="w-4 h-4" />
                  <span>{device.wifi_signal || '--'}</span>
                </div>
                <div className="flex items-center gap-1">
                  <Camera className="w-4 h-4" />
                  <span>{device.fps?.toFixed(0) || '--'} fps</span>
                </div>
              </div>

              {/* Actions */}
              <div className="flex gap-2">
                <button
                  onClick={() => handleSpray(device.device_id)}
                  disabled={device.status !== 'online'}
                  className="flex-1 flex items-center justify-center gap-2 bg-blue-600 text-white py-2 px-4 rounded-lg hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed transition"
                >
                  <Droplets className="w-4 h-4" />
                  Spray
                </button>
              </div>
            </div>
          </div>
        ))}

        {(!devices?.devices || devices.devices.length === 0) && (
          <div className="col-span-full text-center py-12 text-gray-500">
            No cameras registered
          </div>
        )}
      </div>
    </div>
  );
}
