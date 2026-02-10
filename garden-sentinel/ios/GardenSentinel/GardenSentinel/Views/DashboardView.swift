import SwiftUI

struct DashboardView: View {
    @StateObject private var viewModel = DashboardViewModel()
    @EnvironmentObject var alertManager: AlertManager

    var body: some View {
        NavigationView {
            ScrollView {
                VStack(spacing: 20) {
                    // Stats cards
                    LazyVGrid(columns: [
                        GridItem(.flexible()),
                        GridItem(.flexible())
                    ], spacing: 16) {
                        StatCard(
                            title: "Devices Online",
                            value: "\(viewModel.stats?.devicesOnline ?? 0)/\(viewModel.stats?.devicesTotal ?? 0)",
                            icon: "video.fill",
                            color: .green
                        )

                        StatCard(
                            title: "Detections Today",
                            value: "\(viewModel.stats?.detectionsToday ?? 0)",
                            icon: "eye.fill",
                            color: .orange
                        )

                        StatCard(
                            title: "Active Alerts",
                            value: "\(alertManager.unacknowledgedCount)",
                            icon: "bell.fill",
                            color: .red
                        )

                        StatCard(
                            title: "Deterrence Rate",
                            value: String(format: "%.0f%%", (viewModel.stats?.deterrenceSuccessRate ?? 0) * 100),
                            icon: "shield.fill",
                            color: .blue
                        )
                    }
                    .padding(.horizontal)

                    // Weather card
                    if let weather = viewModel.weather {
                        WeatherCard(weather: weather)
                            .padding(.horizontal)
                    }

                    // Recent alerts
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Recent Alerts")
                                .font(.headline)
                            Spacer()
                            NavigationLink("See All") {
                                AlertsView()
                            }
                            .font(.subheadline)
                        }

                        if alertManager.alerts.isEmpty {
                            Text("No recent alerts")
                                .foregroundColor(.secondary)
                                .frame(maxWidth: .infinity, alignment: .center)
                                .padding()
                        } else {
                            ForEach(alertManager.alerts.prefix(5)) { alert in
                                AlertRow(alert: alert)
                            }
                        }
                    }
                    .padding(.horizontal)

                    // Devices status
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Text("Devices")
                                .font(.headline)
                            Spacer()
                            NavigationLink("See All") {
                                CamerasView()
                            }
                            .font(.subheadline)
                        }

                        ForEach(viewModel.devices.prefix(4)) { device in
                            DeviceStatusRow(device: device)
                        }
                    }
                    .padding(.horizontal)
                }
                .padding(.vertical)
            }
            .navigationTitle("Dashboard")
            .refreshable {
                await viewModel.refresh()
                await alertManager.fetchAlerts()
            }
            .onAppear {
                Task {
                    await viewModel.refresh()
                }
                alertManager.startRefreshing()
            }
        }
    }
}

// MARK: - View Model

@MainActor
class DashboardViewModel: ObservableObject {
    @Published var stats: DashboardStats?
    @Published var weather: Weather?
    @Published var devices: [Device] = []
    @Published var isLoading = false

    func refresh() async {
        isLoading = true
        async let statsTask = APIService.shared.getDashboardStats()
        async let weatherTask = APIService.shared.getWeather()
        async let devicesTask = APIService.shared.getDevices()

        do {
            stats = try await statsTask
            weather = try await weatherTask
            devices = try await devicesTask
        } catch {
            print("Dashboard refresh error: \(error)")
        }
        isLoading = false
    }
}

// MARK: - Components

struct StatCard: View {
    let title: String
    let value: String
    let icon: String
    let color: Color

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Image(systemName: icon)
                    .foregroundColor(color)
                Spacer()
            }

            Text(value)
                .font(.title)
                .fontWeight(.bold)

            Text(title)
                .font(.caption)
                .foregroundColor(.secondary)
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }
}

struct WeatherCard: View {
    let weather: Weather

    var body: some View {
        HStack(spacing: 16) {
            Image(systemName: weatherIcon)
                .font(.system(size: 40))
                .foregroundColor(.blue)

            VStack(alignment: .leading, spacing: 4) {
                Text(weather.condition.capitalized)
                    .font(.headline)
                Text("\(Int(weather.temperature))°C • \(Int(weather.humidity))% humidity")
                    .font(.subheadline)
                    .foregroundColor(.secondary)
                Text("Wind: \(Int(weather.windSpeed)) m/s")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(12)
    }

    var weatherIcon: String {
        switch weather.condition.lowercased() {
        case "clear": return "sun.max.fill"
        case "clouds", "cloudy": return "cloud.fill"
        case "rain": return "cloud.rain.fill"
        case "snow": return "cloud.snow.fill"
        case "fog": return "cloud.fog.fill"
        default: return "cloud.fill"
        }
    }
}

struct AlertRow: View {
    let alert: Alert

    var body: some View {
        HStack(spacing: 12) {
            Circle()
                .fill(severityColor)
                .frame(width: 10, height: 10)

            VStack(alignment: .leading, spacing: 2) {
                Text(alert.predatorType?.capitalized ?? "Unknown")
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(alert.timestamp, style: .relative)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if !alert.acknowledged {
                Text("NEW")
                    .font(.caption2)
                    .fontWeight(.bold)
                    .foregroundColor(.white)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 4)
                    .background(Color.red)
                    .cornerRadius(4)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }

    var severityColor: Color {
        switch alert.severity {
        case .low: return .blue
        case .medium: return .yellow
        case .high: return .orange
        case .critical: return .red
        }
    }
}

struct DeviceStatusRow: View {
    let device: Device

    var body: some View {
        HStack(spacing: 12) {
            Image(systemName: deviceIcon)
                .foregroundColor(statusColor)
                .frame(width: 30)

            VStack(alignment: .leading, spacing: 2) {
                Text(device.name)
                    .font(.subheadline)
                    .fontWeight(.medium)

                Text(device.status.rawValue.capitalized)
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Spacer()

            if let battery = device.batteryLevel {
                HStack(spacing: 4) {
                    Image(systemName: batteryIcon(battery))
                    Text("\(battery)%")
                        .font(.caption)
                }
                .foregroundColor(battery < 20 ? .red : .secondary)
            }
        }
        .padding()
        .background(Color(.secondarySystemBackground))
        .cornerRadius(8)
    }

    var deviceIcon: String {
        switch device.type {
        case .camera: return "video.fill"
        case .sprayer: return "drop.fill"
        case .speaker: return "speaker.wave.2.fill"
        case .drone: return "airplane"
        case .sensor: return "sensor.tag.radiowaves.forward.fill"
        }
    }

    var statusColor: Color {
        switch device.status {
        case .online: return .green
        case .offline: return .gray
        case .warning: return .yellow
        case .error: return .red
        }
    }

    func batteryIcon(_ level: Int) -> String {
        switch level {
        case 0..<25: return "battery.25"
        case 25..<50: return "battery.50"
        case 50..<75: return "battery.75"
        default: return "battery.100"
        }
    }
}
