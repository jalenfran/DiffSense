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

// Request interceptor for logging
api.interceptors.request.use(
  (config) => {
    console.log(`API Request: ${config.method?.toUpperCase()} ${config.url}`)
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
  // Repository Management
  async cloneRepository(repoUrl, maxCommits = 50) {
    const response = await api.post('/clone-repository', {
      repo_url: repoUrl,
      max_commits: maxCommits
    })
    return response.data
  },

  async getRepositorySummary(repoId) {
    const response = await api.get(`/repository/${repoId}/summary`)
    return response.data
  },

  async getRepositoryStats(repoId) {
    const response = await api.get(`/repository/${repoId}/stats`)
    return response.data
  },

  async getRepositoryFiles(repoId) {
    const response = await api.get(`/repository/${repoId}/files`)
    return response.data
  },

  // Breaking Change Detection
  async analyzeCommit(repoId, commitHash, includeClaudeAnalysis = true) {
    const response = await api.post(`/analyze-commit/${repoId}`, {
      commit_hash: commitHash,
      include_claude_analysis: includeClaudeAnalysis
    })
    return response.data
  },

  async analyzeCommitRange(repoId, startCommit, endCommit, maxCommits = 20, includeClaudeAnalysis = true) {
    const response = await api.post(`/analyze-commit-range/${repoId}`, {
      start_commit: startCommit,
      end_commit: endCommit,
      max_commits: maxCommits,
      include_claude_analysis: includeClaudeAnalysis
    })
    return response.data
  },

  // RAG System & Intelligent Queries
  async queryRepository(repoId, query, maxResults = 10, includeClaudeResponse = true) {
    const response = await api.post(`/query-repository/${repoId}`, {
      query,
      max_results: maxResults,
      include_claude_response: includeClaudeResponse
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

  // Risk Analysis & Dashboards
  async getRiskDashboard(repoId) {
    const response = await api.get(`/repository/${repoId}/risk-dashboard`)
    return response.data
  }
}

export default api
