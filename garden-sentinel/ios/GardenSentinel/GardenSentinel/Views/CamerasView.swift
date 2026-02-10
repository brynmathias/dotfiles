import SwiftUI

struct CamerasView: View {
    @StateObject private var viewModel = CamerasViewModel()

    var body: some View {
        NavigationView {
            ScrollView {
                LazyVGrid(columns: [
                    GridItem(.flexible()),
                    GridItem(.flexible())
                ], spacing: 16) {
                    ForEach(viewModel.devices) { device in
                        CameraCard(device: device, onSpray: {
                            Task {
                                await viewModel.triggerSpray(deviceId: device.id)
                            }
                        })
                    }
                }
                .padding()
            }
            .navigationTitle("Cameras")
            .refreshable {
                await viewModel.loadDevices()
            }
            .onAppear {
                Task {
                    await viewModel.loadDevices()
                }
            }
        }
    }
}

// MARK: - View Model

@MainActor
class CamerasViewModel: ObservableObject {
    @Published var devices: [Device] = []
    @Published var isLoading = false

    func loadDevices() async {
        isLoading = true
        do {
            devices = try await APIService.shared.getDevices()
        } catch {
            print("Load devices error: \(error)")
        }
        isLoading = false
    }

    func triggerSpray(deviceId: String) async {
        do {
            try await APIService.shared.triggerSpray(deviceId: deviceId)
        } catch {
            print("Spray trigger error: \(error)")
        }
    }
}

// MARK: - Camera Card

struct CameraCard: View {
    let device: Device
    let onSpray: () -> Void

    @State private var showingActions = false

    var body: some View {
        VStack(spacing: 0) {
            // Video placeholder
            ZStack {
                Rectangle()
                    .fill(Color.black)
                    .aspectRatio(16/9, contentMode: .fit)

                if device.status == .online {
                    Image(systemName: "video.fill")
                        .font(.largeTitle)
                        .foregroundColor(.gray)
                } else {
                    VStack(spacing: 8) {
                        Image(systemName: "video.slash.fill")
                            .font(.largeTitle)
                        Text("Offline")
                            .font(.caption)
                    }
                    .foregroundColor(.gray)
                }

                // Status indicator
                VStack {
                    HStack {
                        Spacer()
                        Circle()
                            .fill(statusColor)
                            .frame(width: 12, height: 12)
                            .padding(8)
                    }
                    Spacer()
                }
            }
            .cornerRadius(12, corners: [.topLeft, .topRight])

            // Info bar
            VStack(spacing: 8) {
                HStack {
                    Text(device.name)
                        .font(.subheadline)
                        .fontWeight(.medium)
                        .lineLimit(1)

                    Spacer()

                    Button(action: { showingActions = true }) {
                        Image(systemName: "ellipsis.circle")
                            .foregroundColor(.secondary)
                    }
                }

                HStack(spacing: 12) {
                    if let battery = device.batteryLevel {
                        Label("\(battery)%", systemImage: batteryIcon(battery))
                            .font(.caption)
                            .foregroundColor(battery < 20 ? .red : .secondary)
                    }

                    if let signal = device.signalStrength {
                        Label("\(signal)%", systemImage: "wifi")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    if let fps = device.config?.fps {
                        Label("\(fps) FPS", systemImage: "speedometer")
                            .font(.caption)
                            .foregroundColor(.secondary)
                    }

                    Spacer()
                }
            }
            .padding(12)
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12, corners: [.bottomLeft, .bottomRight])
        }
        .shadow(color: .black.opacity(0.1), radius: 4, x: 0, y: 2)
        .confirmationDialog("Camera Actions", isPresented: $showingActions) {
            Button("Trigger Spray") {
                onSpray()
            }
            Button("Play Sound") {
                // Would trigger sound
            }
            Button("View Full Screen") {
                // Would open full screen view
            }
            Button("Cancel", role: .cancel) {}
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

// MARK: - Corner Radius Extension

extension View {
    func cornerRadius(_ radius: CGFloat, corners: UIRectCorner) -> some View {
        clipShape(RoundedCorner(radius: radius, corners: corners))
    }
}

struct RoundedCorner: Shape {
    var radius: CGFloat = .infinity
    var corners: UIRectCorner = .allCorners

    func path(in rect: CGRect) -> Path {
        let path = UIBezierPath(
            roundedRect: rect,
            byRoundingCorners: corners,
            cornerRadii: CGSize(width: radius, height: radius)
        )
        return Path(path.cgPath)
    }
}
