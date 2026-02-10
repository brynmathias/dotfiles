import SwiftUI

struct LoginView: View {
    @EnvironmentObject var authManager: AuthManager
    @State private var username = ""
    @State private var password = ""
    @State private var serverURL = APIConfig.baseURL
    @State private var showingSettings = false

    var body: some View {
        NavigationView {
            VStack(spacing: 24) {
                // Logo
                VStack(spacing: 8) {
                    Image(systemName: "shield.checkerboard")
                        .font(.system(size: 80))
                        .foregroundColor(.green)

                    Text("Garden Sentinel")
                        .font(.largeTitle)
                        .fontWeight(.bold)

                    Text("Protect your garden")
                        .font(.subheadline)
                        .foregroundColor(.secondary)
                }
                .padding(.top, 40)

                Spacer()

                // Login form
                VStack(spacing: 16) {
                    TextField("Username", text: $username)
                        .textFieldStyle(RoundedTextFieldStyle())
                        .textContentType(.username)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)

                    SecureField("Password", text: $password)
                        .textFieldStyle(RoundedTextFieldStyle())
                        .textContentType(.password)

                    if let error = authManager.errorMessage {
                        Text(error)
                            .font(.caption)
                            .foregroundColor(.red)
                            .multilineTextAlignment(.center)
                    }

                    Button(action: login) {
                        HStack {
                            if authManager.isLoading {
                                ProgressView()
                                    .progressViewStyle(CircularProgressViewStyle(tint: .white))
                            } else {
                                Text("Sign In")
                            }
                        }
                        .frame(maxWidth: .infinity)
                        .padding()
                        .background(Color.green)
                        .foregroundColor(.white)
                        .cornerRadius(12)
                    }
                    .disabled(username.isEmpty || password.isEmpty || authManager.isLoading)
                }
                .padding(.horizontal, 32)

                Spacer()

                // Server settings
                Button(action: { showingSettings = true }) {
                    HStack {
                        Image(systemName: "server.rack")
                        Text("Server: \(serverURL)")
                            .font(.caption)
                    }
                    .foregroundColor(.secondary)
                }
                .padding(.bottom, 20)
            }
            .background(Color(.systemBackground))
            .sheet(isPresented: $showingSettings) {
                ServerSettingsSheet(serverURL: $serverURL)
            }
        }
    }

    private func login() {
        Task {
            await authManager.login(username: username, password: password)
        }
    }
}

struct RoundedTextFieldStyle: TextFieldStyle {
    func _body(configuration: TextField<Self._Label>) -> some View {
        configuration
            .padding()
            .background(Color(.secondarySystemBackground))
            .cornerRadius(12)
    }
}

struct ServerSettingsSheet: View {
    @Binding var serverURL: String
    @Environment(\.dismiss) var dismiss
    @State private var tempURL: String = ""

    var body: some View {
        NavigationView {
            Form {
                Section(header: Text("Server URL")) {
                    TextField("https://garden-sentinel.local", text: $tempURL)
                        .autocapitalization(.none)
                        .disableAutocorrection(true)
                        .keyboardType(.URL)
                }

                Section(footer: Text("Enter the URL of your Garden Sentinel server")) {
                    Button("Save") {
                        serverURL = tempURL
                        UserDefaults.standard.set(tempURL, forKey: "api_base_url")
                        dismiss()
                    }
                    .disabled(tempURL.isEmpty)
                }
            }
            .navigationTitle("Server Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
            }
            .onAppear {
                tempURL = serverURL
            }
        }
    }
}
