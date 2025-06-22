import { createContext, useContext, useState, useEffect } from 'react'
import { diffSenseAPI } from '../services/api'

const RepositoryContext = createContext()

export const useRepository = () => {
  const context = useContext(RepositoryContext)
  if (!context) {
    throw new Error('useRepository must be used within a RepositoryProvider')
  }
  return context
}

export const RepositoryProvider = ({ children }) => {
  const [selectedRepo, setSelectedRepo] = useState(null)
  const [repoId, setRepoId] = useState(null)
  const [repoStats, setRepoStats] = useState(null)
  const [repoFiles, setRepoFiles] = useState([])
  const [riskDashboard, setRiskDashboard] = useState(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState(null)

  // Chat state
  const [messages, setMessages] = useState([])
  const [selectedFiles, setSelectedFiles] = useState([])
  const [isChatLoading, setIsChatLoading] = useState(false)

  // Clear state when repo changes
  useEffect(() => {
    if (selectedRepo) {
      setMessages([])
      setSelectedFiles([])
      setRepoStats(null)
      setRepoFiles([])
      setRiskDashboard(null)
      setError(null)
    }
  }, [selectedRepo])

  const selectRepository = async (repo) => {
    setSelectedRepo(repo)
    setIsLoading(true)
    setError(null)

    try {
      // First, clone/analyze the repository
      const repoUrl = repo.clone_url || repo.html_url
      console.log('Cloning repository:', repoUrl)
      
      const cloneResult = await diffSenseAPI.cloneRepository(repoUrl)
      setRepoId(cloneResult.repo_id)

      // Fetch repository data in parallel
      const [stats, files, dashboard] = await Promise.all([
        diffSenseAPI.getRepositoryStats(cloneResult.repo_id).catch(err => {
          console.warn('Failed to fetch repo stats:', err)
          return null
        }),        diffSenseAPI.getRepositoryFiles(cloneResult.repo_id).catch(err => {
          console.warn('Failed to fetch repo files:', err)
          return []
        }),        diffSenseAPI.getRiskDashboard(cloneResult.repo_id).catch(err => {
          console.warn('Failed to fetch risk dashboard:', err)
          return null
        })
      ])

      setRepoStats(stats)
      
      // Ensure files is always an array
      const safeFiles = Array.isArray(files) ? files : (files?.files || [])
      setRepoFiles(safeFiles)
      
      setRiskDashboard(dashboard)

    } catch (error) {
      console.error('Error selecting repository:', error)
      setError(error.response?.data?.detail || error.message || 'Failed to analyze repository')
    } finally {
      setIsLoading(false)
    }
  }

  const sendChatMessage = async (message, contextFiles = []) => {
    if (!repoId) {
      throw new Error('No repository selected')
    }

    // Add user message
    const userMessage = {
      content: message,
      isUser: true,
      timestamp: Date.now()
    }
    setMessages(prev => [...prev, userMessage])
    setIsChatLoading(true)

    try {
      // Query the repository
      const response = await diffSenseAPI.queryRepository(
        repoId, 
        message, 
        10, 
        true
      )

      // Add AI response
      const aiMessage = {
        content: {
          type: 'query_response',
          response: response.response,
          confidence: response.confidence,
          sources: response.sources,
          claude_enhanced: response.claude_enhanced
        },
        isUser: false,
        timestamp: Date.now()
      }
      setMessages(prev => [...prev, aiMessage])

    } catch (error) {
      console.error('Error sending chat message:', error)
      
      // Add error message
      const errorMessage = {
        content: `Sorry, I encountered an error: ${error.response?.data?.detail || error.message}`,
        isUser: false,
        timestamp: Date.now()
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setIsChatLoading(false)
    }
  }

  const analyzeCommit = async (commitHash) => {
    if (!repoId) {
      throw new Error('No repository selected')
    }

    setIsChatLoading(true)

    try {
      const analysis = await diffSenseAPI.analyzeCommit(repoId, commitHash, true)
      
      // Add analysis to chat
      const analysisMessage = {
        content: {
          type: 'commit_analysis',
          ...analysis
        },
        isUser: false,
        timestamp: Date.now()
      }
      setMessages(prev => [...prev, analysisMessage])

      return analysis
    } catch (error) {
      console.error('Error analyzing commit:', error)
      throw error
    } finally {
      setIsChatLoading(false)
    }
  }

  const searchRepositoryContent = async (query, type = 'files') => {
    if (!repoId) {
      throw new Error('No repository selected')
    }

    try {
      if (type === 'files') {
        return await diffSenseAPI.searchFiles(repoId, query)
      } else if (type === 'commits') {
        return await diffSenseAPI.searchCommits(repoId, query)
      }
    } catch (error) {
      console.error('Error searching repository:', error)
      throw error
    }
  }

  const clearChat = () => {
    setMessages([])
  }

  const value = {
    // Repository state
    selectedRepo,
    repoId,
    repoStats,
    repoFiles,
    riskDashboard,
    isLoading,
    error,

    // Chat state
    messages,
    selectedFiles,
    isChatLoading,

    // Actions
    selectRepository,
    sendChatMessage,
    analyzeCommit,
    searchRepositoryContent,
    clearChat,
    setSelectedFiles
  }

  return (
    <RepositoryContext.Provider value={value}>
      {children}
    </RepositoryContext.Provider>
  )
}
