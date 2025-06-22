import axios from 'axios'
import { config } from '../config'

// Configure base API URL
const API_BASE_URL = config.API_BASE_URL

const api = axios.create({
    baseURL: API_BASE_URL,
    timeout: config.API_TIMEOUT,
    headers: {
        'Content-Type': 'application/json',
    },
})

// Auth token management
let sessionToken = null

// Initialize session token from localStorage
const initializeSessionToken = () => {
    const storedToken = localStorage.getItem('diffsense-session-token')
    if (storedToken) {
        sessionToken = storedToken
        console.log('Session token loaded from localStorage')
    }
}

// Initialize on module load
initializeSessionToken()

// Request interceptor for authentication and logging
api.interceptors.request.use(
    (config) => {
        console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)

        // Always check localStorage for the latest token
        if (!sessionToken) {
            sessionToken = localStorage.getItem('diffsense-session-token')
        }    // Add session token to authenticated requests
        // Only exclude the initial OAuth endpoints that don't require auth
        const isOAuthInitEndpoint = config.url === '/auth/github' || config.url === '/auth/github/callback'

        if (sessionToken && !isOAuthInitEndpoint) {
            config.headers.Authorization = `Bearer ${sessionToken}`
            console.log('Adding auth header with session token')
        } else if (!sessionToken && !isOAuthInitEndpoint) {
            console.log('No session token available for authenticated request')
        }

        return config
    },
    (error) => {
        console.error('API Request Error:', error)
        return Promise.reject(error)
    }
)

// Response interceptor for error handling
api.interceptors.response.use(
    (response) => {
        console.log(`API Response: ${response.status} ${response.config.url}`)
        return response
    },
    (error) => {
        console.error('API Response Error:', error.response?.data || error.message)

        // Handle authentication errors
        if (error.response?.status === 401) {
            console.error('Authentication failed - clearing session')
            sessionToken = null
            localStorage.removeItem('diffsense-session-token')
            // Optionally redirect to login or refresh the page
            window.location.reload()
        }

        // Handle network errors when connecting to external server
        if (error.code === 'ECONNABORTED') {
            console.error('Request timeout - API server may be unavailable')
        } else if (error.code === 'ERR_NETWORK') {
            console.error('Network error - Check if API server is accessible')
        } else if (error.response?.status === 0) {
            console.error('CORS error - Check proxy configuration and server CORS settings')
        }

        return Promise.reject(error)
    }
)

export const diffSenseAPI = {
    // Authentication
    async initiateGitHubAuth() {
        const response = await api.get('/auth/github')
        return response.data
    },

    async handleGitHubCallback(code, state) {
        const response = await api.post('/auth/github/callback', {
            code,
            state
        })

        if (response.data.session_id) {
            sessionToken = response.data.session_id
            localStorage.setItem('diffsense-session-token', sessionToken)
        }

        return response.data
    },
    async getCurrentUser() {
        const response = await api.get('/auth/user')
        return response.data
    },
    async getUserRepositories() {
        const response = await api.get('/user/repositories')
        return response.data
    },

    async getGitHubRepositories() {
        const response = await api.get('/auth/github/repositories')
        return response.data
    },

    async logout() {
        try {
            await api.post('/auth/logout')
        } finally {
            sessionToken = null
            localStorage.removeItem('diffsense-session-token')
        }
    },
    // Repository Management
    async cloneRepository(repoUrl, useAuth = false) {
        console.log('Cloning repository:', { repoUrl, useAuth, hasToken: !!sessionToken })
        const response = await api.post('/clone-repository', {
            repo_url: repoUrl,
            use_auth: useAuth
        })
        return response.data
    },    async getRepositoryFiles(repoId, options = {}) {
        const params = {}
        if (options.all_files) params.all_files = true
        
        const response = await api.get(`/repository/${repoId}/files`, { params })
        return response.data
    },

    async getRepositoryStats(repoId) {
        const response = await api.get(`/repository/${repoId}/stats`)
        return response.data
    },    async getRepositoryCommits(repoId, limit = 50) {
        const response = await api.get(`/repository/${repoId}/commits`, {
            params: { limit }
        })
        return response.data.commits || []
    },

    async cleanupRepository(repoId) {
        const response = await api.delete(`/repository/${repoId}`)
        return response.data
    },

    // Breaking Change Analysis
    async analyzeCommit(repoId, commitHash, includeClaudeAnalysis = true) {
        const response = await api.post(`/analyze-commit/${repoId}`, {
            commit_hash: commitHash,
            include_claude_analysis: includeClaudeAnalysis
        })
        return response.data
    },

    async analyzeCommitRange(repoId, startCommit, endCommit, maxCommits = 100, includeClaudeAnalysis = false) {
        const response = await api.post(`/analyze-commit-range/${repoId}`, {
            start_commit: startCommit,
            end_commit: endCommit,
            max_commits: maxCommits,
            include_claude_analysis: includeClaudeAnalysis
        })
        return response.data
    },

    // Advanced Breaking Change Analysis
    async analyzeBreakingChanges(repoId, options = {}) {
        try {
            console.log('Analyzing breaking changes:', { repoId, options })
            
            const response = await api.post(`/repository/${repoId}/analyze-breaking-changes`, options)
            
            if (response.data) {
                console.log('Breaking change analysis completed:', response.data)
                return response.data
            }
            
            throw new Error('No analysis data received')
        } catch (error) {
            console.error('Error analyzing breaking changes:', error)
            throw error
        }
    },

    // RAG Queries & Intelligent Search
    async queryRepository(repoId, query, maxResults = 10) {
        const response = await api.post(`/query-repository/${repoId}`, {
            query,
            max_results: maxResults
        })
        return response.data
    },

    async enhancedRepositoryQuery(repoId, query, maxResults = 10) {
        const response = await api.post(`/repository/${repoId}/query/enhanced`, {
            query,
            max_results: maxResults
        })
        return response.data
    },

    async searchCommits(repoId, query, maxResults = 10) {
        const response = await api.get(`/repository/${repoId}/commits/search`, {
            params: { query, max_results: maxResults }
        })
        return response.data
    },

    async searchFiles(repoId, query, maxResults = 10) {
        const response = await api.get(`/repository/${repoId}/files/search`, {
            params: { query, max_results: maxResults }
        })
        return response.data
    },

    // Advanced Insights
    async getRepositorySummary(repoId) {
        const response = await api.get(`/repository/${repoId}/summary`)
        return response.data
    },

    async getRiskDashboard(repoId) {
        const response = await api.get(`/repository/${repoId}/risk-dashboard`)
        return response.data
    },

    async getCommitAnalysis(repoId, commitHash) {
        const response = await api.get(`/repository/${repoId}/commit/${commitHash}/analysis`)
        return response.data
    },

    async getFileInsights(repoId, filePath) {
        const response = await api.get(`/repository/${repoId}/file/${encodeURIComponent(filePath)}/insights`)
        return response.data
    },

    async getSemanticDriftAnalysis(repoId, days = 30) {
        const response = await api.get(`/repository/${repoId}/insights/semantic-drift`, {
            params: { days }
        })
        return response.data
    },

    // Chat System (Persistent)
    async createChat(repoId = null, title = 'New Chat') {
        const response = await api.post('/chats', {
            repo_id: repoId,
            title
        })
        return response.data
    },

    async listChats(includeArchived = false) {
        const response = await api.get('/chats', {
            params: { include_archived: includeArchived }
        })
        return response.data
    },

    async getChat(chatId) {
        const response = await api.get(`/chats/${chatId}`)
        return response.data
    },

    async sendChatMessage(chatId, message) {
        const response = await api.post(`/chats/${chatId}/messages`, {
            chat_id: chatId,
            message
        })
        return response.data
    },

    async updateChatTitle(chatId, title) {
        const response = await api.put(`/chats/${chatId}`, {
            title
        })
        return response.data
    },

    async deleteChat(chatId) {
        const response = await api.delete(`/chats/${chatId}`)
        return response.data
    },    async archiveChat(chatId) {
        const response = await api.post(`/chats/${chatId}/archive`)
        return response.data
    },

    // File Content Endpoints
    async getFileContent(repoId, filePath, options = {}) {
        const params = new URLSearchParams()
        if (options.commitHash) params.append('commit_hash', options.commitHash)
        if (options.showDiff) params.append('show_diff', 'true')
        
        const response = await api.get(`/repository/${repoId}/file/${encodeURIComponent(filePath)}?${params}`)
        return response.data
    },

    async getFileHistory(repoId, filePath, options = {}) {
        const params = new URLSearchParams()
        if (options.limit) params.append('limit', options.limit)
        if (options.showDiffs) params.append('show_diffs', 'true')
        
        const response = await api.get(`/repository/${repoId}/file-history/${encodeURIComponent(filePath)}?${params}`)
        return response.data
    },    // Commit Information Endpoints
    async getCommitFiles(repoId, commitHash, options = {}) {
        const params = new URLSearchParams()
        if (options.includeDiffStats || options.include_diff_stats) params.append('include_diff_stats', 'true')
        if (options.includeMetadata || options.include_metadata) params.append('include_metadata', 'true')
        
        const response = await api.get(`/repository/${repoId}/commit/${commitHash}/files?${params}`)
        return response.data
    },

    async compareCommits(repoId, baseCommit, headCommit, options = {}) {
        const params = new URLSearchParams()
        if (options.filePath) params.append('file_path', options.filePath)
        if (options.contextLines) params.append('context_lines', options.contextLines)
        
        const response = await api.get(`/repository/${repoId}/compare/${baseCommit}...${headCommit}?${params}`)
        return response.data
    },

    // Utility functions
    setSessionToken(token) {
        sessionToken = token
        if (token) {
            localStorage.setItem('diffsense-session-token', token)
            console.log('Session token saved to localStorage and memory')
        } else {
            localStorage.removeItem('diffsense-session-token')
            console.log('Session token cleared from localStorage and memory')
        }
    },

    getSessionToken() {
        // Always check localStorage for the latest token
        if (!sessionToken) {
            sessionToken = localStorage.getItem('diffsense-session-token')
        }
        return sessionToken
    },
    isAuthenticated() {
        // Always check localStorage for the latest token
        if (!sessionToken) {
            sessionToken = localStorage.getItem('diffsense-session-token')
        }
        return !!sessionToken
    },

    // Debug function to check auth state
    debugAuthState() {
        const localStorageToken = localStorage.getItem('diffsense-session-token')
        console.log('Auth Debug:', {
            memoryToken: sessionToken,
            localStorageToken: localStorageToken,
            isAuthenticated: !!sessionToken || !!localStorageToken
        })
        return {
            memoryToken: sessionToken,
            localStorageToken: localStorageToken,
            isAuthenticated: !!sessionToken || !!localStorageToken
        }
    }
}

// Expose for debugging
if (typeof window !== 'undefined') {
    window.diffSenseAPI = diffSenseAPI
}

export default api
