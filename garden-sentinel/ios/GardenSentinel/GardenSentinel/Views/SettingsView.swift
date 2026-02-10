import SwiftUI

struct SettingsView: View {
    @EnvironmentObject var authManager: AuthManager
    @AppStorage("notifications_enabled") private var notificationsEnabled = true
    @AppStorage("critical_only_notifications") private var criticalOnly = false
    @AppStorage("api_base_url") private var serverURL = "https://garden-sentinel.local"

    @State private var showingLogoutConfirmation = false

    var body: some View {
        NavigationView {
            Form {
                // User section
                Section {
                    if let user = authManager.currentUser {
                        HStack {
                            Image(systemName: "person.circle.fill")
                                .font(.system(size: 50))
                                .foregroundColor(.green)

                            VStack(alignment: .leading, spacing: 4) {
                                Text(user.username)
                                    .font(.headline)
                                Text(user.role.rawValue.capitalized)
                                    .font(.caption)
                                    .foregroundColor(.secondary)
                            }
                        }
                        .padding(.vertical, 8)
                    }
                }

                // Notifications section
                Section(header: Text("Notifications")) {
                    Toggle("Enable Notifications", isOn: $notificationsEnabled)

                    Toggle("Critical Alerts Only", isOn: $criticalOnly)
                        .disabled(!notificationsEnabled)

                    NavigationLink("Notification Sounds") {
                        NotificationSoundsView()
                    }
                }

                // Server section
                Section(header: Text("Server")) {
                    HStack {
                        Text("Server URL")
                        Spacer()
                        Text(serverURL)
                            .foregroundColor(.secondary)
                            .lineLimit(1)
                    }

                    NavigationLink("Change Server") {
                        ServerConfigView(serverURL: $serverURL)
                    }
                }

                // Sites section
                Section(header: Text("Sites")) {
                    NavigationLink("Manage Sites") {
                        SitesManagementView()
                    }
                }

                // About section
                Section(header: Text("About")) {
                    HStack {
                        Text("Version")
                        Spacer()
                        Text("1.0.0")
                            .foregroundColor(.secondary)
                    }

                    Link(destination: URL(string: "https://github.com/garden-sentinel/garden-sentinel")!) {
                        HStack {
                            Text("GitHub")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                        }
                    }

                    Link(destination: URL(string: "https://garden-sentinel.docs")!) {
                        HStack {
                            Text("Documentation")
                            Spacer()
                            Image(systemName: "arrow.up.right.square")
                                .foregroundColor(.secondary)
                        }
                    }
                }

                // Logout section
                Section {
                    Button(role: .destructive) {
                        showingLogoutConfirmation = true
                    } label: {
                        HStack {
                            Spacer()
                            Text("Sign Out")
                            Spacer()
                        }
                    }
                }
            }
            .navigationTitle("Settings")
            .confirmationDialog(
                "Sign Out",
                isPresented: $showingLogoutConfirmation,
                titleVisibility: .visible
            ) {
                Button("Sign Out", role: .destructive) {
                    Task {
                        await authManager.logout()
                    }
                }
                Button("Cancel", role: .cancel) {}
            } message: {
                Text("Are you sure you want to sign out?")
            }
        }
    }
}

// MARK: - Sub Views

struct NotificationSoundsView: View {
    @AppStorage("alert_sound") private var alertSound = "default"

    let sounds = ["default", "alarm", "chime", "bell", "urgent"]

    var body: some View {
        Form {
            Section(header: Text("Alert Sound")) {
                ForEach(sounds, id: \.self) { sound in
                    Button {
                        alertSound = sound
                    } label: {
                        HStack {
                            Text(sound.capitalized)
                                .foregroundColor(.primary)
                            Spacer()
                            if alertSound == sound {
                                Image(systemName: "checkmark")
                                    .foregroundColor(.green)
                            }
                        }
                    }
                }
            }
        }
        .navigationTitle("Notification Sounds")
        .navigationBarTitleDisplayMode(.inline)
    }
}

struct ServerConfigView: View {
    @Binding var serverURL: String
    @State private var tempURL: String = ""
    @State private var isTestingConnection = false
    @State private var connectionStatus: ConnectionStatus?
    @Environment(\.dismiss) var dismiss

    enum ConnectionStatus {
        case success, failure(String)
    }

    var body: some View {
        Form {
            Section(header: Text("Server URL")) {
                TextField("https://garden-sentinel.local", text: $tempURL)
                    .autocapitalization(.none)
                    .disableAutocorrection(true)
                    .keyboardType(.URL)
            }

            Section {
                Button {
                    testConnection()
                } label: {
                    HStack {
                        Text("Test Connection")
                        Spacer()
                        if isTestingConnection {
                            ProgressView()
                        } else if let status = connectionStatus {
                            switch status {
                            case .success:
                                Image(systemName: "checkmark.circle.fill")
                                    .foregroundColor(.green)
                            case .failure:
                                Image(systemName: "xmark.circle.fill")
                                    .foregroundColor(.red)
                            }
                        }
                    }
                }
                .disabled(tempURL.isEmpty || isTestingConnection)

                if case .failure(let message) = connectionStatus {
                    Text(message)
                        .font(.caption)
                        .foregroundColor(.red)
                }
            }

            Section {
                Button("Save") {
                    serverURL = tempURL
                    dismiss()
                }
                .disabled(tempURL.isEmpty)
            }
        }
        .navigationTitle("Server Configuration")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            tempURL = serverURL
        }
    }

    private func testConnection() {
        isTestingConnection = true
        connectionStatus = nil

        // Test connection by trying to reach the server
        guard let url = URL(string: "\(tempURL)/api/health") else {
            connectionStatus = .failure("Invalid URL")
            isTestingConnection = false
            return
        }

        URLSession.shared.dataTask(with: url) { _, response, error in
            DispatchQueue.main.async {
                isTestingConnection = false
                if let error = error {
                    connectionStatus = .failure(error.localizedDescription)
                } else if let httpResponse = response as? HTTPURLResponse,
                          (200...299).contains(httpResponse.statusCode) {
                    connectionStatus = .success
                } else {
                    connectionStatus = .failure("Server not responding")
                }
            }
        }.resume()
    }
}

struct SitesManagementView: View {
    @State private var sites: [Site] = []
    @State private var isLoading = false

    var body: some View {
        Group {
            if isLoading {
                ProgressView()
            } else if sites.isEmpty {
                VStack(spacing: 16) {
                    Image(systemName: "house")
                        .font(.system(size: 60))
                        .foregroundColor(.secondary)
                    Text("No sites configured")
                        .font(.headline)
                        .foregroundColor(.secondary)
                }
            } else {
                List(sites) { site in
                    HStack {
                        VStack(alignment: .leading, spacing: 4) {
                            Text(site.name)
                                .font(.headline)
                            Text(site.location.address ?? "No address")
                                .font(.caption)
                                .foregroundColor(.secondary)
                        }

                        Spacer()

                        Circle()
                            .fill(site.status == .online ? Color.green : Color.gray)
                            .frame(width: 10, height: 10)
                    }
                }
            }
        }
        .navigationTitle("Sites")
        .navigationBarTitleDisplayMode(.inline)
        .onAppear {
            loadSites()
        }
    }

    private func loadSites() {
        isLoading = true
        Task {
            do {
                sites = try await APIService.shared.getSites()
            } catch {
                print("Load sites error: \(error)")
            }
            isLoading = false
        }
    }
}
