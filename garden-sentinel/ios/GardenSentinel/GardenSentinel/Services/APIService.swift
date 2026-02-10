import Foundation
import Combine

// MARK: - API Configuration

struct APIConfig {
    static var baseURL: String {
        UserDefaults.standard.string(forKey: "api_base_url") ?? "https://garden-sentinel.local"
    }

    static var token: String? {
        get { KeychainHelper.load(key: "auth_token") }
        set {
            if let value = newValue {
                KeychainHelper.save(key: "auth_token", value: value)
            } else {
                KeychainHelper.delete(key: "auth_token")
            }
        }
    }
}

// MARK: - API Error

enum APIError: LocalizedError {
    case invalidURL
    case invalidResponse
    case httpError(Int)
    case decodingError(Error)
    case networkError(Error)
    case unauthorized
    case notFound
    case serverError(String)

    var errorDescription: String? {
        switch self {
        case .invalidURL: return "Invalid URL"
        case .invalidResponse: return "Invalid server response"
        case .httpError(let code): return "HTTP error: \(code)"
        case .decodingError(let error): return "Decoding error: \(error.localizedDescription)"
        case .networkError(let error): return "Network error: \(error.localizedDescription)"
        case .unauthorized: return "Unauthorized - please login again"
        case .notFound: return "Resource not found"
        case .serverError(let message): return message
        }
    }
}

// MARK: - API Service

class APIService {
    static let shared = APIService()

    private let session: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    private init() {
        let config = URLSessionConfiguration.default
        config.timeoutIntervalForRequest = 30
        config.timeoutIntervalForResource = 60
        self.session = URLSession(configuration: config)

        self.decoder = JSONDecoder()
        self.decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let value = try container.decode(Double.self)
            return Date(timeIntervalSince1970: value)
        }

        self.encoder = JSONEncoder()
    }

    // MARK: - Generic Request

    func request<T: Decodable>(
        endpoint: String,
        method: String = "GET",
        body: Encodable? = nil,
        queryParams: [String: String]? = nil
    ) async throws -> T {
        var urlString = "\(APIConfig.baseURL)\(endpoint)"

        if let params = queryParams, !params.isEmpty {
            let queryString = params.map { "\($0.key)=\($0.value)" }.joined(separator: "&")
            urlString += "?\(queryString)"
        }

        guard let url = URL(string: urlString) else {
            throw APIError.invalidURL
        }

        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        if let token = APIConfig.token {
            request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization")
        }

        if let body = body {
            request.httpBody = try encoder.encode(AnyEncodable(body))
        }

        do {
            let (data, response) = try await session.data(for: request)

            guard let httpResponse = response as? HTTPURLResponse else {
                throw APIError.invalidResponse
            }

            switch httpResponse.statusCode {
            case 200...299:
                do {
                    return try decoder.decode(T.self, from: data)
                } catch {
                    throw APIError.decodingError(error)
                }
            case 401:
                throw APIError.unauthorized
            case 404:
                throw APIError.notFound
            default:
                if let errorMessage = try? JSONDecoder().decode([String: String].self, from: data),
                   let detail = errorMessage["detail"] {
                    throw APIError.serverError(detail)
                }
                throw APIError.httpError(httpResponse.statusCode)
            }
        } catch let error as APIError {
            throw error
        } catch {
            throw APIError.networkError(error)
        }
    }

    // MARK: - Auth

    func login(username: String, password: String) async throws -> AuthResponse {
        struct LoginRequest: Encodable {
            let username: String
            let password: String
        }

        let response: AuthResponse = try await request(
            endpoint: "/api/auth/login",
            method: "POST",
            body: LoginRequest(username: username, password: password)
        )

        APIConfig.token = response.accessToken
        return response
    }

    func logout() async throws {
        let _: EmptyResponse = try await request(endpoint: "/api/auth/logout", method: "POST")
        APIConfig.token = nil
    }

    func getCurrentUser() async throws -> User {
        try await request(endpoint: "/api/auth/me")
    }

    // MARK: - Devices

    func getDevices() async throws -> [Device] {
        try await request(endpoint: "/api/devices")
    }

    func getDevice(id: String) async throws -> Device {
        try await request(endpoint: "/api/devices/\(id)")
    }

    // MARK: - Alerts

    func getAlerts(limit: Int = 50, acknowledged: Bool? = nil) async throws -> [Alert] {
        var params: [String: String] = ["limit": String(limit)]
        if let ack = acknowledged {
            params["acknowledged"] = String(ack)
        }
        return try await request(endpoint: "/api/alerts", queryParams: params)
    }

    func acknowledgeAlert(id: String) async throws -> Alert {
        try await request(endpoint: "/api/alerts/\(id)/acknowledge", method: "POST")
    }

    // MARK: - Detections

    func getDetections(limit: Int = 50) async throws -> [Detection] {
        try await request(endpoint: "/api/detections", queryParams: ["limit": String(limit)])
    }

    // MARK: - Zones

    func getZones() async throws -> [Zone] {
        try await request(endpoint: "/api/zones")
    }

    // MARK: - Weather

    func getWeather() async throws -> Weather {
        try await request(endpoint: "/api/weather/current")
    }

    func getActivityPredictions() async throws -> [ActivityPrediction] {
        try await request(endpoint: "/api/weather/activity")
    }

    // MARK: - Dashboard

    func getDashboardStats() async throws -> DashboardStats {
        try await request(endpoint: "/api/dashboard/stats")
    }

    // MARK: - Sites

    func getSites() async throws -> [Site] {
        try await request(endpoint: "/api/sites")
    }

    // MARK: - Predator Profiles

    func getPredatorProfiles() async throws -> [PredatorProfile] {
        try await request(endpoint: "/api/predators")
    }

    // MARK: - Neighbor Network

    func getNeighbors() async throws -> [Neighbor] {
        try await request(endpoint: "/api/network/neighbors")
    }

    func getSharedAlerts() async throws -> [SharedAlert] {
        try await request(endpoint: "/api/network/alerts")
    }

    // MARK: - Actions

    func triggerSpray(deviceId: String, duration: Double = 2.0) async throws {
        struct SprayRequest: Encodable {
            let duration: Double
        }
        let _: EmptyResponse = try await request(
            endpoint: "/api/devices/\(deviceId)/spray",
            method: "POST",
            body: SprayRequest(duration: duration)
        )
    }

    func playSound(deviceId: String, soundType: String) async throws {
        struct SoundRequest: Encodable {
            let soundType: String
            enum CodingKeys: String, CodingKey {
                case soundType = "sound_type"
            }
        }
        let _: EmptyResponse = try await request(
            endpoint: "/api/devices/\(deviceId)/sound",
            method: "POST",
            body: SoundRequest(soundType: soundType)
        )
    }
}

// MARK: - Helpers

struct EmptyResponse: Decodable {}

struct AnyEncodable: Encodable {
    private let encode: (Encoder) throws -> Void

    init<T: Encodable>(_ value: T) {
        self.encode = value.encode
    }

    func encode(to encoder: Encoder) throws {
        try encode(encoder)
    }
}

// MARK: - Keychain Helper

class KeychainHelper {
    static func save(key: String, value: String) {
        guard let data = value.data(using: .utf8) else { return }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecValueData as String: data
        ]

        SecItemDelete(query as CFDictionary)
        SecItemAdd(query as CFDictionary, nil)
    }

    static func load(key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true
        ]

        var result: AnyObject?
        SecItemCopyMatching(query as CFDictionary, &result)

        guard let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    static func delete(key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrAccount as String: key
        ]
        SecItemDelete(query as CFDictionary)
    }
}
