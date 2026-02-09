import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { settingsApi, usersApi, apiKeysApi } from '../lib/api';
import { useState } from 'react';
import { useAuthStore } from '../stores/authStore';
import { Save, Plus, Trash2, Key, User, Bell, Shield, Volume2 } from 'lucide-react';

export default function Settings() {
  const { user } = useAuthStore();
  const queryClient = useQueryClient();
  const [activeTab, setActiveTab] = useState('general');

  const { data: settings } = useQuery({
    queryKey: ['settings'],
    queryFn: () => settingsApi.get().then((res) => res.data),
  });

  const { data: users } = useQuery({
    queryKey: ['users'],
    queryFn: () => usersApi.list().then((res) => res.data),
    enabled: user?.role === 'admin',
  });

  const { data: apiKeys } = useQuery({
    queryKey: ['api-keys'],
    queryFn: () => apiKeysApi.list().then((res) => res.data),
    enabled: user?.role === 'admin',
  });

  const updateSettingsMutation = useMutation({
    mutationFn: (data: any) => settingsApi.update(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['settings'] });
    },
  });

  const createApiKeyMutation = useMutation({
    mutationFn: (data: { name: string; device_id: string }) => apiKeysApi.create(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const deleteApiKeyMutation = useMutation({
    mutationFn: (keyId: string) => apiKeysApi.delete(keyId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['api-keys'] });
    },
  });

  const tabs = [
    { id: 'general', label: 'General', icon: Settings },
    { id: 'detection', label: 'Detection', icon: Shield },
    { id: 'alerts', label: 'Alerts', icon: Bell },
    { id: 'audio', label: 'Audio', icon: Volume2 },
    ...(user?.role === 'admin' ? [{ id: 'users', label: 'Users', icon: User }] : []),
    ...(user?.role === 'admin' ? [{ id: 'api-keys', label: 'API Keys', icon: Key }] : []),
  ];

  return (
    <div className="p-6">
      <h1 className="text-2xl font-bold mb-6">Settings</h1>

      <div className="flex gap-6">
        {/* Sidebar */}
        <div className="w-48 flex-shrink-0">
          <nav className="space-y-1">
            {tabs.map((tab) => (
              <button
                key={tab.id}
                onClick={() => setActiveTab(tab.id)}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg ${
                  activeTab === tab.id
                    ? 'bg-green-100 text-green-800'
                    : 'text-gray-700 hover:bg-gray-100'
                }`}
              >
                <tab.icon className="w-4 h-4" />
                {tab.label}
              </button>
            ))}
          </nav>
        </div>

        {/* Content */}
        <div className="flex-1 bg-white rounded-lg shadow p-6">
          {activeTab === 'general' && (
            <GeneralSettings settings={settings} onSave={updateSettingsMutation.mutate} />
          )}
          {activeTab === 'detection' && (
            <DetectionSettings settings={settings} onSave={updateSettingsMutation.mutate} />
          )}
          {activeTab === 'alerts' && (
            <AlertSettings settings={settings} onSave={updateSettingsMutation.mutate} />
          )}
          {activeTab === 'audio' && (
            <AudioSettings settings={settings} onSave={updateSettingsMutation.mutate} />
          )}
          {activeTab === 'users' && <UsersSettings users={users} />}
          {activeTab === 'api-keys' && (
            <ApiKeysSettings
              apiKeys={apiKeys}
              onCreate={createApiKeyMutation.mutate}
              onDelete={deleteApiKeyMutation.mutate}
            />
          )}
        </div>
      </div>
    </div>
  );
}

function GeneralSettings({ settings, onSave }: { settings: any; onSave: (data: any) => void }) {
  const [formData, setFormData] = useState(settings?.general || {});

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">General Settings</h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Site Name</label>
          <input
            type="text"
            value={formData.site_name || ''}
            onChange={(e) => setFormData({ ...formData, site_name: e.target.value })}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Timezone</label>
          <select
            value={formData.timezone || 'UTC'}
            onChange={(e) => setFormData({ ...formData, timezone: e.target.value })}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value="UTC">UTC</option>
            <option value="America/New_York">Eastern Time</option>
            <option value="America/Chicago">Central Time</option>
            <option value="America/Denver">Mountain Time</option>
            <option value="America/Los_Angeles">Pacific Time</option>
            <option value="Europe/London">London</option>
            <option value="Europe/Paris">Paris</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Data Retention (days)
          </label>
          <input
            type="number"
            value={formData.retention_days || 30}
            onChange={(e) => setFormData({ ...formData, retention_days: parseInt(e.target.value) })}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          />
        </div>
      </div>

      <button
        onClick={() => onSave({ general: formData })}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
      >
        <Save className="w-4 h-4" />
        Save Changes
      </button>
    </div>
  );
}

function DetectionSettings({ settings, onSave }: { settings: any; onSave: (data: any) => void }) {
  const [formData, setFormData] = useState(settings?.detection || {});

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Detection Settings</h2>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Confidence Threshold
          </label>
          <input
            type="range"
            min="0.1"
            max="1"
            step="0.05"
            value={formData.confidence_threshold || 0.7}
            onChange={(e) =>
              setFormData({ ...formData, confidence_threshold: parseFloat(e.target.value) })
            }
            className="w-full"
          />
          <span className="text-sm text-gray-500">
            {((formData.confidence_threshold || 0.7) * 100).toFixed(0)}%
          </span>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Enabled Predators</label>
          <div className="space-y-2">
            {['fox', 'hawk', 'coyote', 'raccoon', 'cat', 'snake'].map((predator) => (
              <label key={predator} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={formData.enabled_predators?.includes(predator) ?? true}
                  onChange={(e) => {
                    const current = formData.enabled_predators || [
                      'fox',
                      'hawk',
                      'coyote',
                      'raccoon',
                      'cat',
                      'snake',
                    ];
                    setFormData({
                      ...formData,
                      enabled_predators: e.target.checked
                        ? [...current, predator]
                        : current.filter((p: string) => p !== predator),
                    });
                  }}
                  className="rounded text-green-600 focus:ring-green-500"
                />
                <span className="capitalize">{predator}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={formData.weather_adjustment ?? true}
              onChange={(e) => setFormData({ ...formData, weather_adjustment: e.target.checked })}
              className="rounded text-green-600 focus:ring-green-500"
            />
            <span className="text-sm font-medium text-gray-700">
              Auto-adjust sensitivity based on weather
            </span>
          </label>
        </div>
      </div>

      <button
        onClick={() => onSave({ detection: formData })}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
      >
        <Save className="w-4 h-4" />
        Save Changes
      </button>
    </div>
  );
}

function AlertSettings({ settings, onSave }: { settings: any; onSave: (data: any) => void }) {
  const [formData, setFormData] = useState(settings?.alerts || {});

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Alert Settings</h2>

      <div className="space-y-4">
        <div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={formData.push_notifications ?? true}
              onChange={(e) => setFormData({ ...formData, push_notifications: e.target.checked })}
              className="rounded text-green-600 focus:ring-green-500"
            />
            <span className="text-sm font-medium text-gray-700">Push Notifications</span>
          </label>
        </div>

        <div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={formData.email_alerts ?? false}
              onChange={(e) => setFormData({ ...formData, email_alerts: e.target.checked })}
              className="rounded text-green-600 focus:ring-green-500"
            />
            <span className="text-sm font-medium text-gray-700">Email Alerts</span>
          </label>
        </div>

        {formData.email_alerts && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Email Address</label>
            <input
              type="email"
              value={formData.email_address || ''}
              onChange={(e) => setFormData({ ...formData, email_address: e.target.value })}
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            />
          </div>
        )}

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Minimum Severity for Alerts
          </label>
          <select
            value={formData.min_severity || 'medium'}
            onChange={(e) => setFormData({ ...formData, min_severity: e.target.value })}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          >
            <option value="low">Low</option>
            <option value="medium">Medium</option>
            <option value="high">High</option>
            <option value="critical">Critical</option>
          </select>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Quiet Hours (no notifications)
          </label>
          <div className="flex items-center gap-2">
            <input
              type="time"
              value={formData.quiet_start || '22:00'}
              onChange={(e) => setFormData({ ...formData, quiet_start: e.target.value })}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            />
            <span>to</span>
            <input
              type="time"
              value={formData.quiet_end || '06:00'}
              onChange={(e) => setFormData({ ...formData, quiet_end: e.target.value })}
              className="px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            />
          </div>
        </div>
      </div>

      <button
        onClick={() => onSave({ alerts: formData })}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
      >
        <Save className="w-4 h-4" />
        Save Changes
      </button>
    </div>
  );
}

function AudioSettings({ settings, onSave }: { settings: any; onSave: (data: any) => void }) {
  const [formData, setFormData] = useState(settings?.audio || {});

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-semibold">Audio Deterrent Settings</h2>

      <div className="space-y-4">
        <div>
          <label className="flex items-center gap-2">
            <input
              type="checkbox"
              checked={formData.enabled ?? true}
              onChange={(e) => setFormData({ ...formData, enabled: e.target.checked })}
              className="rounded text-green-600 focus:ring-green-500"
            />
            <span className="text-sm font-medium text-gray-700">Enable Audio Deterrents</span>
          </label>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Master Volume</label>
          <input
            type="range"
            min="0"
            max="100"
            value={formData.volume || 80}
            onChange={(e) => setFormData({ ...formData, volume: parseInt(e.target.value) })}
            className="w-full"
          />
          <span className="text-sm text-gray-500">{formData.volume || 80}%</span>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Enabled Sounds</label>
          <div className="space-y-2">
            {[
              { id: 'dog_bark', label: 'Dog Barks' },
              { id: 'fox_distress', label: 'Fox Distress Calls' },
              { id: 'predator_growl', label: 'Predator Growls' },
              { id: 'ultrasonic', label: 'Ultrasonic' },
              { id: 'alarm', label: 'Alarm Siren' },
              { id: 'tts', label: 'Voice Warnings' },
            ].map((sound) => (
              <label key={sound.id} className="flex items-center gap-2">
                <input
                  type="checkbox"
                  checked={formData.enabled_sounds?.includes(sound.id) ?? true}
                  onChange={(e) => {
                    const current = formData.enabled_sounds || [
                      'dog_bark',
                      'fox_distress',
                      'predator_growl',
                      'ultrasonic',
                      'alarm',
                      'tts',
                    ];
                    setFormData({
                      ...formData,
                      enabled_sounds: e.target.checked
                        ? [...current, sound.id]
                        : current.filter((s: string) => s !== sound.id),
                    });
                  }}
                  className="rounded text-green-600 focus:ring-green-500"
                />
                <span>{sound.label}</span>
              </label>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
            Cooldown Between Plays (seconds)
          </label>
          <input
            type="number"
            value={formData.cooldown || 30}
            onChange={(e) => setFormData({ ...formData, cooldown: parseInt(e.target.value) })}
            className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
          />
        </div>
      </div>

      <button
        onClick={() => onSave({ audio: formData })}
        className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700"
      >
        <Save className="w-4 h-4" />
        Save Changes
      </button>
    </div>
  );
}

function UsersSettings({ users }: { users: any }) {
  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">User Management</h2>
        <button className="flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700">
          <Plus className="w-4 h-4" />
          Add User
        </button>
      </div>

      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Username
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Role
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Created
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {(users || []).map((user: any) => (
            <tr key={user.id}>
              <td className="px-4 py-3 text-sm font-medium text-gray-900">{user.username}</td>
              <td className="px-4 py-3 text-sm text-gray-500 capitalize">{user.role}</td>
              <td className="px-4 py-3 text-sm text-gray-500">
                {new Date(user.created_at).toLocaleDateString()}
              </td>
              <td className="px-4 py-3 text-right">
                <button className="text-red-600 hover:text-red-800">
                  <Trash2 className="w-4 h-4" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function ApiKeysSettings({
  apiKeys,
  onCreate,
  onDelete,
}: {
  apiKeys: any;
  onCreate: (data: { name: string; device_id: string }) => void;
  onDelete: (keyId: string) => void;
}) {
  const [showCreate, setShowCreate] = useState(false);
  const [newKey, setNewKey] = useState({ name: '', device_id: '' });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold">API Keys</h2>
        <button
          onClick={() => setShowCreate(true)}
          className="flex items-center gap-2 px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700"
        >
          <Plus className="w-4 h-4" />
          Create API Key
        </button>
      </div>

      {showCreate && (
        <div className="p-4 bg-gray-50 rounded-lg space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Name</label>
            <input
              type="text"
              value={newKey.name}
              onChange={(e) => setNewKey({ ...newKey, name: e.target.value })}
              placeholder="e.g., North Camera Pi"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Device ID</label>
            <input
              type="text"
              value={newKey.device_id}
              onChange={(e) => setNewKey({ ...newKey, device_id: e.target.value })}
              placeholder="e.g., camera-north-01"
              className="w-full px-3 py-2 border rounded-lg focus:ring-2 focus:ring-green-500"
            />
          </div>
          <div className="flex gap-2">
            <button
              onClick={() => {
                onCreate(newKey);
                setShowCreate(false);
                setNewKey({ name: '', device_id: '' });
              }}
              className="px-3 py-1.5 bg-green-600 text-white text-sm rounded-lg hover:bg-green-700"
            >
              Create
            </button>
            <button
              onClick={() => setShowCreate(false)}
              className="px-3 py-1.5 bg-gray-200 text-gray-700 text-sm rounded-lg hover:bg-gray-300"
            >
              Cancel
            </button>
          </div>
        </div>
      )}

      <table className="min-w-full divide-y divide-gray-200">
        <thead className="bg-gray-50">
          <tr>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Name
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Device ID
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Key Prefix
            </th>
            <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">
              Last Used
            </th>
            <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 uppercase">
              Actions
            </th>
          </tr>
        </thead>
        <tbody className="divide-y divide-gray-200">
          {(apiKeys || []).map((key: any) => (
            <tr key={key.id}>
              <td className="px-4 py-3 text-sm font-medium text-gray-900">{key.name}</td>
              <td className="px-4 py-3 text-sm text-gray-500">{key.device_id}</td>
              <td className="px-4 py-3 text-sm text-gray-500 font-mono">{key.prefix}...</td>
              <td className="px-4 py-3 text-sm text-gray-500">
                {key.last_used ? new Date(key.last_used).toLocaleString() : 'Never'}
              </td>
              <td className="px-4 py-3 text-right">
                <button
                  onClick={() => onDelete(key.id)}
                  className="text-red-600 hover:text-red-800"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
