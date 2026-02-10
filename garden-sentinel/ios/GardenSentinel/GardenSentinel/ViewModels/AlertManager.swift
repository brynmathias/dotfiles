import Foundation
import UserNotifications

@MainActor
class AlertManager: ObservableObject {
    static let shared = AlertManager()

    @Published var alerts: [Alert] = []
    @Published var isLoading = false
    @Published var errorMessage: String?

    var unacknowledgedCount: Int {
        alerts.filter { !$0.acknowledged }.count
    }

    private var refreshTask: Task<Void, Never>?

    private init() {
        requestNotificationPermission()
    }

    func startRefreshing() {
        refreshTask?.cancel()
        refreshTask = Task {
            while !Task.isCancelled {
                await fetchAlerts()
                try? await Task.sleep(nanoseconds: 30_000_000_000) // 30 seconds
            }
        }
    }

    func stopRefreshing() {
        refreshTask?.cancel()
        refreshTask = nil
    }

    func fetchAlerts() async {
        isLoading = true
        do {
            let newAlerts = try await APIService.shared.getAlerts(limit: 100)

            // Check for new unacknowledged alerts
            let newUnack = newAlerts.filter { newAlert in
                !newAlert.acknowledged &&
                !alerts.contains(where: { $0.id == newAlert.id })
            }

            for alert in newUnack {
                sendLocalNotification(for: alert)
            }

            alerts = newAlerts
            errorMessage = nil
        } catch {
            errorMessage = error.localizedDescription
        }
        isLoading = false
    }

    func acknowledgeAlert(_ alert: Alert) async {
        do {
            let updated = try await APIService.shared.acknowledgeAlert(id: alert.id)
            if let index = alerts.firstIndex(where: { $0.id == alert.id }) {
                alerts[index] = updated
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func requestNotificationPermission() {
        UNUserNotificationCenter.current().requestAuthorization(options: [.alert, .sound, .badge]) { granted, error in
            if granted {
                print("Notification permission granted")
            }
        }
    }

    private func sendLocalNotification(for alert: Alert) {
        let content = UNMutableNotificationContent()
        content.title = "Garden Sentinel Alert"
        content.body = "\(alert.severity.rawValue.capitalized): \(alert.predatorType ?? "Unknown predator") detected"
        content.sound = alert.severity == .critical ? .defaultCritical : .default

        let request = UNNotificationRequest(
            identifier: alert.id,
            content: content,
            trigger: nil
        )

        UNUserNotificationCenter.current().add(request)
    }
}
