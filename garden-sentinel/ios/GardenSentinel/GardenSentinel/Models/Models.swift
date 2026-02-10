import Foundation
import CoreLocation

// MARK: - User & Auth

struct User: Codable, Identifiable {
    let id: String
    let username: String
    let role: UserRole
    let createdAt: Date?

    enum CodingKeys: String, CodingKey {
        case id, username, role
        case createdAt = "created_at"
    }
}

enum UserRole: String, Codable {
    case viewer, operator, admin
}

struct AuthResponse: Codable {
    let accessToken: String
    let tokenType: String
    let user: User

    enum CodingKeys: String, CodingKey {
        case accessToken = "access_token"
        case tokenType = "token_type"
        case user
    }
}

// MARK: - Devices

struct Device: Codable, Identifiable {
    let id: String
    let name: String
    let type: DeviceType
    let status: DeviceStatus
    let lastSeen: Date?
    let batteryLevel: Int?
    let signalStrength: Int?
    let config: DeviceConfig?

    enum CodingKeys: String, CodingKey {
        case id, name, type, status, config
        case lastSeen = "last_seen"
        case batteryLevel = "battery_level"
        case signalStrength = "signal_strength"
    }
}

enum DeviceType: String, Codable {
    case camera, sprayer, speaker, drone, sensor
}

enum DeviceStatus: String, Codable {
    case online, offline, warning, error
}

struct DeviceConfig: Codable {
    let resolution: [Int]?
    let fps: Int?
    let deterrents: [String]?
}

// MARK: - Detections & Alerts

struct Detection: Codable, Identifiable {
    let id: String
    let deviceId: String
    let timestamp: Date
    let predatorType: String
    let confidence: Double
    let bbox: [Int]?
    let framePath: String?

    enum CodingKeys: String, CodingKey {
        case id, timestamp, bbox
        case deviceId = "device_id"
        case predatorType = "predator_type"
        case confidence
        case framePath = "frame_path"
    }
}

struct Alert: Codable, Identifiable {
    let id: String
    let detectionId: String?
    let deviceId: String
    let timestamp: Date
    let severity: AlertSeverity
    let predatorType: String?
    let confidence: Double?
    let acknowledged: Bool
    let acknowledgedBy: String?
    let acknowledgedAt: Date?

    enum CodingKeys: String, CodingKey {
        case id, timestamp, severity, acknowledged
        case detectionId = "detection_id"
        case deviceId = "device_id"
        case predatorType = "predator_type"
        case confidence
        case acknowledgedBy = "acknowledged_by"
        case acknowledgedAt = "acknowledged_at"
    }

    var severityColor: String {
        switch severity {
        case .low: return "blue"
        case .medium: return "yellow"
        case .high: return "orange"
        case .critical: return "red"
        }
    }
}

enum AlertSeverity: String, Codable {
    case low, medium, high, critical
}

// MARK: - Zones

struct Zone: Codable, Identifiable {
    let id: String
    let name: String
    let zoneType: ZoneType
    let geometry: GeoJSON
    let properties: ZoneProperties?

    enum CodingKeys: String, CodingKey {
        case id, name, geometry, properties
        case zoneType = "zone_type"
    }
}

enum ZoneType: String, Codable {
    case protected, monitored, exclusion, safe
}

struct ZoneProperties: Codable {
    let priority: String?
    let deterrenceLevel: String?

    enum CodingKeys: String, CodingKey {
        case priority
        case deterrenceLevel = "deterrence_level"
    }
}

struct GeoJSON: Codable {
    let type: String
    let coordinates: [[[Double]]]
}

// MARK: - Weather

struct Weather: Codable {
    let temperature: Double
    let humidity: Double
    let windSpeed: Double
    let windDirection: Double
    let condition: String
    let visibility: Double
    let precipitation: Double

    enum CodingKeys: String, CodingKey {
        case temperature, humidity, condition, visibility, precipitation
        case windSpeed = "wind_speed"
        case windDirection = "wind_direction"
    }
}

struct ActivityPrediction: Codable {
    let predatorType: String
    let activityLevel: Double
    let factors: [String]

    enum CodingKeys: String, CodingKey {
        case predatorType = "predator_type"
        case activityLevel = "activity_level"
        case factors
    }
}

// MARK: - Analytics

struct DashboardStats: Codable {
    let devicesOnline: Int
    let devicesTotal: Int
    let detectionsToday: Int
    let alertsUnacknowledged: Int
    let deterrenceSuccessRate: Double

    enum CodingKeys: String, CodingKey {
        case devicesOnline = "devices_online"
        case devicesTotal = "devices_total"
        case detectionsToday = "detections_today"
        case alertsUnacknowledged = "alerts_unacknowledged"
        case deterrenceSuccessRate = "deterrence_success_rate"
    }
}

// MARK: - Sites

struct Site: Codable, Identifiable {
    let id: String
    let name: String
    let location: SiteLocation
    let status: SiteStatus
    let deviceIds: [String]

    enum CodingKeys: String, CodingKey {
        case id, name, location, status
        case deviceIds = "device_ids"
    }
}

struct SiteLocation: Codable {
    let lat: Double
    let lng: Double
    let address: String?
    let timezone: String
}

enum SiteStatus: String, Codable {
    case online, offline, degraded, maintenance
}

// MARK: - Predator Profiles

struct PredatorProfile: Codable, Identifiable {
    let id: String
    let predatorType: String
    let name: String?
    let firstSeen: Date
    let lastSeen: Date
    let sightingCount: Int
    let threatLevel: ThreatLevel

    enum CodingKeys: String, CodingKey {
        case id, name
        case predatorType = "predator_type"
        case firstSeen = "first_seen"
        case lastSeen = "last_seen"
        case sightingCount = "sighting_count"
        case threatLevel = "threat_level"
    }
}

enum ThreatLevel: String, Codable {
    case low, medium, high, critical
}

// MARK: - Neighbor Network

struct Neighbor: Codable, Identifiable {
    let id: String
    let name: String
    let distanceKm: Double?
    let trustLevel: String
    let alertsReceived: Int
    let alertsSent: Int

    enum CodingKeys: String, CodingKey {
        case id, name
        case distanceKm = "distance_km"
        case trustLevel = "trust_level"
        case alertsReceived = "alerts_received"
        case alertsSent = "alerts_sent"
    }
}

struct SharedAlert: Codable, Identifiable {
    let id: String
    let sourceId: String
    let predatorType: String
    let confidence: Double
    let timestamp: Date
    let location: Location
    let heading: Double?
    let expiresAt: Date

    enum CodingKeys: String, CodingKey {
        case id, confidence, timestamp, location, heading
        case sourceId = "source_id"
        case predatorType = "predator_type"
        case expiresAt = "expires_at"
    }
}

struct Location: Codable {
    let lat: Double
    let lng: Double

    var coordinate: CLLocationCoordinate2D {
        CLLocationCoordinate2D(latitude: lat, longitude: lng)
    }
}
