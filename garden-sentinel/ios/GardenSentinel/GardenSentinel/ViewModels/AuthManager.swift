import Foundation
import Combine

@MainActor
class AuthManager: ObservableObject {
    static let shared = AuthManager()

    @Published var isAuthenticated = false
    @Published var currentUser: User?
    @Published var isLoading = false
    @Published var errorMessage: String?

    private init() {
        // Check for existing token
        if APIConfig.token != nil {
            Task {
                await checkAuthentication()
            }
        }
    }

    func checkAuthentication() async {
        guard APIConfig.token != nil else {
            isAuthenticated = false
            return
        }

        isLoading = true
        do {
            currentUser = try await APIService.shared.getCurrentUser()
            isAuthenticated = true
        } catch {
            // Token invalid, clear it
            APIConfig.token = nil
            isAuthenticated = false
        }
        isLoading = false
    }

    func login(username: String, password: String) async -> Bool {
        isLoading = true
        errorMessage = nil

        do {
            let response = try await APIService.shared.login(username: username, password: password)
            currentUser = response.user
            isAuthenticated = true
            isLoading = false
            return true
        } catch let error as APIError {
            errorMessage = error.errorDescription
            isLoading = false
            return false
        } catch {
            errorMessage = error.localizedDescription
            isLoading = false
            return false
        }
    }

    func logout() async {
        do {
            try await APIService.shared.logout()
        } catch {
            // Ignore logout errors, clear local state anyway
        }

        APIConfig.token = nil
        currentUser = nil
        isAuthenticated = false
    }
}
