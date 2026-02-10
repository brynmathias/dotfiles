import SwiftUI
import MapKit

struct MapView: View {
    @StateObject private var viewModel = MapViewModel()

    var body: some View {
        NavigationView {
            ZStack {
                Map(coordinateRegion: $viewModel.region, annotationItems: viewModel.annotations) { item in
                    MapAnnotation(coordinate: item.coordinate) {
                        AnnotationView(annotation: item)
                    }
                }
                .ignoresSafeArea(edges: .bottom)

                // Legend
                VStack {
                    Spacer()
                    LegendView()
                        .padding()
                }
            }
            .navigationTitle("Map")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .navigationBarTrailing) {
                    Button(action: { viewModel.centerOnGarden() }) {
                        Image(systemName: "location.fill")
                    }
                }
            }
            .onAppear {
                Task {
                    await viewModel.loadData()
                }
            }
        }
    }
}

// MARK: - View Model

@MainActor
class MapViewModel: ObservableObject {
    @Published var region = MKCoordinateRegion(
        center: CLLocationCoordinate2D(latitude: 51.5074, longitude: -0.1278),
        span: MKCoordinateSpan(latitudeDelta: 0.005, longitudeDelta: 0.005)
    )
    @Published var annotations: [MapAnnotationItem] = []
    @Published var zones: [Zone] = []
    @Published var devices: [Device] = []

    func loadData() async {
        do {
            async let zonesTask = APIService.shared.getZones()
            async let devicesTask = APIService.shared.getDevices()

            zones = try await zonesTask
            devices = try await devicesTask

            updateAnnotations()
        } catch {
            print("Map load error: \(error)")
        }
    }

    func updateAnnotations() {
        var items: [MapAnnotationItem] = []

        // Add device annotations
        for device in devices {
            // In real app, devices would have location data
            // Using placeholder positions for demo
            items.append(MapAnnotationItem(
                id: device.id,
                coordinate: region.center,
                type: .camera,
                title: device.name,
                status: device.status
            ))
        }

        annotations = items
    }

    func centerOnGarden() {
        withAnimation {
            region = MKCoordinateRegion(
                center: region.center,
                span: MKCoordinateSpan(latitudeDelta: 0.002, longitudeDelta: 0.002)
            )
        }
    }
}

// MARK: - Annotation

struct MapAnnotationItem: Identifiable {
    let id: String
    let coordinate: CLLocationCoordinate2D
    let type: AnnotationType
    let title: String
    let status: DeviceStatus

    enum AnnotationType {
        case camera, zone, detection, track
    }
}

struct AnnotationView: View {
    let annotation: MapAnnotationItem

    var body: some View {
        VStack(spacing: 0) {
            ZStack {
                Circle()
                    .fill(statusColor)
                    .frame(width: 36, height: 36)

                Image(systemName: iconName)
                    .foregroundColor(.white)
                    .font(.system(size: 16))
            }

            Image(systemName: "triangle.fill")
                .font(.system(size: 10))
                .foregroundColor(statusColor)
                .rotationEffect(.degrees(180))
                .offset(y: -3)
        }
    }

    var iconName: String {
        switch annotation.type {
        case .camera: return "video.fill"
        case .zone: return "square.dashed"
        case .detection: return "exclamationmark.triangle.fill"
        case .track: return "pawprint.fill"
        }
    }

    var statusColor: Color {
        switch annotation.status {
        case .online: return .green
        case .offline: return .gray
        case .warning: return .yellow
        case .error: return .red
        }
    }
}

struct LegendView: View {
    var body: some View {
        HStack(spacing: 16) {
            LegendItem(color: .green, label: "Online")
            LegendItem(color: .gray, label: "Offline")
            LegendItem(color: .red, label: "Alert")
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 8)
        .background(.ultraThinMaterial)
        .cornerRadius(8)
    }
}

struct LegendItem: View {
    let color: Color
    let label: String

    var body: some View {
        HStack(spacing: 4) {
            Circle()
                .fill(color)
                .frame(width: 8, height: 8)
            Text(label)
                .font(.caption)
        }
    }
}
