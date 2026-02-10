import SwiftUI

struct AlertsView: View {
    @EnvironmentObject var alertManager: AlertManager
    @State private var filter: AlertFilter = .all

    var filteredAlerts: [Alert] {
        switch filter {
        case .all:
            return alertManager.alerts
        case .unacknowledged:
            return alertManager.alerts.filter { !$0.acknowledged }
        case .critical:
            return alertManager.alerts.filter { $0.severity == .critical }
        }
    }

    var body: some View {
        NavigationView {
            VStack(spacing: 0) {
                // Filter picker
                Picker("Filter", selection: $filter) {
                    Text("All").tag(AlertFilter.all)
                    Text("Unread").tag(AlertFilter.unacknowledged)
                    Text("Critical").tag(AlertFilter.critical)
                }
                .pickerStyle(.segmented)
                .padding()

                if filteredAlerts.isEmpty {
                    Spacer()
                    VStack(spacing: 16) {
                        Image(systemName: "bell.slash")
                            .font(.system(size: 60))
                            .foregroundColor(.secondary)
                        Text("No alerts")
                            .font(.headline)
                            .foregroundColor(.secondary)
                    }
                    Spacer()
                } else {
                    List {
                        ForEach(filteredAlerts) { alert in
                            AlertListRow(alert: alert) {
                                Task {
                                    await alertManager.acknowledgeAlert(alert)
                                }
                            }
                        }
                    }
                    .listStyle(.plain)
                }
            }
            .navigationTitle("Alerts")
            .refreshable {
                await alertManager.fetchAlerts()
            }
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    if alertManager.unacknowledgedCount > 0 {
                        Button("Acknowledge All") {
                            acknowledgeAll()
                        }
                    }
                }
            }
        }
    }

    private func acknowledgeAll() {
        Task {
            for alert in alertManager.alerts.filter({ !$0.acknowledged }) {
                await alertManager.acknowledgeAlert(alert)
            }
        }
    }
}

enum AlertFilter: String, CaseIterable {
    case all, unacknowledged, critical
}

struct AlertListRow: View {
    let alert: Alert
    let onAcknowledge: () -> Void

    var body: some View {
        HStack(spacing: 12) {
            // Severity indicator
            RoundedRectangle(cornerRadius: 4)
                .fill(severityColor)
                .frame(width: 4)

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text(alert.predatorType?.capitalized ?? "Unknown predator")
                        .font(.headline)

                    Spacer()

                    Text(alert.severity.rawValue.capitalized)
                        .font(.caption)
                        .fontWeight(.medium)
                        .foregroundColor(.white)
                        .padding(.horizontal, 8)
                        .padding(.vertical, 4)
                        .background(severityColor)
                        .cornerRadius(4)
                }

                HStack {
                    Image(systemName: "video.fill")
                        .font(.caption)
                    Text(alert.deviceId)
                        .font(.caption)
                        .foregroundColor(.secondary)

                    Spacer()

                    Text(alert.timestamp, style: .relative)
                        .font(.caption)
                        .foregroundColor(.secondary)
                }

                if let confidence = alert.confidence {
                    HStack {
                        Text("Confidence:")
                            .font(.caption)
                            .foregroundColor(.secondary)
                        Text(String(format: "%.0f%%", confidence * 100))
                            .font(.caption)
                            .fontWeight(.medium)
                    }
                }
            }

            if !alert.acknowledged {
                Button(action: onAcknowledge) {
                    Image(systemName: "checkmark.circle")
                        .font(.title2)
                        .foregroundColor(.green)
                }
                .buttonStyle(.plain)
            } else {
                Image(systemName: "checkmark.circle.fill")
                    .font(.title2)
                    .foregroundColor(.gray)
            }
        }
        .padding(.vertical, 8)
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
