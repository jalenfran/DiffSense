// API Configuration
export const config = {
  API_BASE_URL: import.meta.env.VITE_API_BASE_URL || 'http://76.125.217.28:8080/api',
  AUTH_BASE_URL: import.meta.env.VITE_AUTH_BASE_URL || 'http://localhost:3000',
  API_TIMEOUT: 30000, // 30 seconds
  
  // Default values for API calls
  DEFAULT_MAX_COMMITS: 50,
  DEFAULT_MAX_RESULTS: 10,
  
  // Risk level thresholds
  RISK_LEVELS: {
    LOW: 0.3,
    MEDIUM: 0.6,
    HIGH: 0.8
  }
}

export default config
