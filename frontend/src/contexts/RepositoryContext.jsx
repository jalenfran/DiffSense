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
 
    // Chat state - now using persistent chat system
    const [chats, setChats] = useState([])
    const [currentChatId, setCurrentChatId] = useState(null)
    const [currentChatMessages, setCurrentChatMessages] = useState([])
    const [selectedFiles, setSelectedFiles] = useState([])
    const [isChatLoading, setIsChatLoading] = useState(false)    // Clear state when repo changes
    useEffect(() => {
        if (selectedRepo) {
            setRepoStats(null)
            setRepoFiles([])
            setRiskDashboard(null)
            setError(null)
            setCurrentChatId(null)
            setCurrentChatMessages([])
            setSelectedFiles([])
            loadChatsForRepo()
        }
    }, [selectedRepo])

    // Load current chat messages when chat changes
    useEffect(() => {
        if (currentChatId) {
            loadChatMessages(currentChatId)
        } else {
            setCurrentChatMessages([])
        }
    }, [currentChatId])

    const selectRepository = async (repo, existingRepoId = null) => {
        setSelectedRepo(repo)
        setIsLoading(true)
        setError(null)

        try {
            // Use the repo_id from the repo object (set when added to DiffSense)
            // or the explicitly passed existingRepoId
            let repoId = existingRepoId || repo.repo_id
            
            if (!repoId) {
                // This should only happen for repositories that haven't been properly added to DiffSense
                throw new Error('Repository not properly added to DiffSense. Please add it first.')
            }

            setRepoId(repoId)// Fetch repository data in parallel
            const [stats, files, dashboard] = await Promise.all([                diffSenseAPI.getRepositoryStats(repoId).catch(err => {
                    console.warn('Failed to fetch repo stats:', err)
                    return null
                }),
                diffSenseAPI.getRepositoryFiles(repoId).catch(err => {
                    console.warn('Failed to fetch repo files:', err)
                    return []
                }),
                diffSenseAPI.getRiskDashboard(repoId).catch(err => {
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

    const loadChatsForRepo = async () => {
        if (!diffSenseAPI.isAuthenticated() || !selectedRepo) return

        try {
            const allChats = await diffSenseAPI.listChats()
            
            // Filter chats for current repo using the repo identifier
            // The repo_id in chats uses underscore format (owner_repo), but our repo uses slash format (owner/repo)
            const repoIdentifier = selectedRepo.id || selectedRepo.full_name
            const repoIdentifierUnderscore = repoIdentifier.replace('/', '_')
            
            const repoChats = allChats.filter(chat => 
                chat.repo_id === repoIdentifier || 
                chat.repo_id === repoIdentifierUnderscore
            )

            setChats(repoChats)

            // If there are existing chats, select the most recent one
            if (repoChats.length > 0) {
                const mostRecentChat = repoChats.sort((a, b) =>
                    new Date(b.updated_at) - new Date(a.updated_at)
                )[0]
                setCurrentChatId(mostRecentChat.id)
            } else {
                // No existing chats, clear current chat
                setCurrentChatId(null)
                setCurrentChatMessages([])
            }
        } catch (error) {
            console.error('Error loading chats:', error)
            setChats([])
        }
    }

    const loadChatMessages = async (chatId) => {
        try {
            const chatData = await diffSenseAPI.getChat(chatId)
            const apiMessages = chatData.messages || []

            // Transform API messages into UI format
            const uiMessages = []
            apiMessages.forEach(msg => {
                // Add user message
                uiMessages.push({
                    content: msg.message,
                    isUser: true,
                    timestamp: msg.timestamp,
                    id: `${msg.id}_user`
                })

                // Add assistant response
                uiMessages.push({
                    content: msg.response,
                    isUser: false,
                    timestamp: msg.timestamp,
                    id: `${msg.id}_assistant`
                })
            })

            setCurrentChatMessages(uiMessages)
        } catch (error) {
            console.error('Error loading chat messages:', error)
            setCurrentChatMessages([])
        }
    }

    const createNewChat = async (title = null) => {
        if (!diffSenseAPI.isAuthenticated()) {
            throw new Error('Authentication required for chat')
        }

        try {
            const chatTitle = title || `${selectedRepo?.name || 'Repository'} Chat`
            const newChat = await diffSenseAPI.createChat(repoId, chatTitle)

            setChats(prev => [newChat, ...prev])
            setCurrentChatId(newChat.id)

            return newChat
        } catch (error) {
            console.error('Error creating chat:', error)
            throw error
        }
    }

    const sendChatMessage = async (message, contextFiles = []) => {
        if (!diffSenseAPI.isAuthenticated()) {
            throw new Error('Authentication required for chat')
        }

        // Create a chat if none exists
        let chatId = currentChatId
        if (!chatId) {
            const newChat = await createNewChat()
            chatId = newChat.id
        }

        // Add user message immediately to the UI
        const userMessage = {
            content: message,
            isUser: true,
            timestamp: new Date().toISOString(),
            id: `temp_${Date.now()}`
        }
        setCurrentChatMessages(prev => [...prev, userMessage])

        setIsChatLoading(true)

        try {
            // Send message and get response
            const response = await diffSenseAPI.sendChatMessage(chatId, message)

            // Reload messages to get the latest state (this will replace the temp message)
            await loadChatMessages(chatId)

            return response

        } catch (error) {
            console.error('Error sending chat message:', error)
            // Remove the temporary user message on error
            setCurrentChatMessages(prev => prev.filter(msg => msg.id !== userMessage.id))
            throw error
        } finally {
            setIsChatLoading(false)
        }
    }

    const switchChat = (chatId) => {
        setCurrentChatId(chatId)
    }

    const selectChat = (chatId) => {
        setCurrentChatId(chatId)
        // Messages will be loaded by the useEffect that watches currentChatId
    }

    const deleteChat = async (chatId) => {
        try {
            await diffSenseAPI.deleteChat(chatId)
            setChats(prev => prev.filter(chat => chat.id !== chatId))

            if (currentChatId === chatId) {
                setCurrentChatId(null)
            }
        } catch (error) {
            console.error('Error deleting chat:', error)
            throw error
        }
    }

    const updateChatTitle = async (chatId, newTitle) => {
        try {
            await diffSenseAPI.updateChatTitle(chatId, newTitle)
            setChats(prev => prev.map(chat =>
                chat.id === chatId ? { ...chat, title: newTitle } : chat
            ))
        } catch (error) {
            console.error('Error updating chat title:', error)
            throw error
        }
    }

    const analyzeCommit = async (commitHash) => {
        if (!repoId) {
            throw new Error('No repository selected')
        }

        try {
            const analysis = await diffSenseAPI.analyzeCommit(repoId, commitHash, true)
            return analysis
        } catch (error) {
            console.error('Error analyzing commit:', error)
            throw error
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
        chats,
        currentChatId,
        messages: currentChatMessages, // Backward compatibility
        selectedFiles,
        isChatLoading,        // Actions
        selectRepository,
        sendChatMessage,
        createNewChat,
        selectChat,
        switchChat,
        deleteChat,
        updateChatTitle,
        analyzeCommit,
        searchRepositoryContent,
        setSelectedFiles
    }

    return (
        <RepositoryContext.Provider value={value}>
            {children}
        </RepositoryContext.Provider>
    )
}
