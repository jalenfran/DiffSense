import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import MainContent from './MainContent'
import AddRepositoryDialog from './AddRepositoryDialog'
import { useRepository } from '../contexts/RepositoryContext'
import { diffSenseAPI } from '../services/api'

function Dashboard({ user, onLogout }) {
    const [repositories, setRepositories] = useState([])
    const [favorites, setFavorites] = useState(() => {
        const saved = localStorage.getItem('diffsense-favorites')
        return saved ? JSON.parse(saved) : []
    })
    const [recentlyViewed, setRecentlyViewed] = useState(() => {
        const saved = localStorage.getItem('diffsense-recently-viewed')
        return saved ? JSON.parse(saved) : []
    })
    const [searchQuery, setSearchQuery] = useState('')
    const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false)
    const [showAddDialog, setShowAddDialog] = useState(false)
    const [isLoadingRepos, setIsLoadingRepos] = useState(false)

    const { selectRepository, selectedRepo, isLoading: isRepositoryLoading } = useRepository()

    useEffect(() => {
        localStorage.setItem('diffsense-favorites', JSON.stringify(favorites))
    }, [favorites])

    useEffect(() => {
        localStorage.setItem('diffsense-recently-viewed', JSON.stringify(recentlyViewed))
    }, [recentlyViewed])

    // Load analyzed repositories from DiffSense platform (not localStorage)
    useEffect(() => {
        if (user && diffSenseAPI.isAuthenticated()) {
            loadAnalyzedRepositories()
        }
    }, [user])

    const loadAnalyzedRepositories = async () => {
        setIsLoadingRepos(true)
        try {
            const analyzedRepos = await diffSenseAPI.getUserRepositories()

            // Convert DiffSense API repository format to UI format
            const formattedRepos = analyzedRepos.map(repo => {
                // Extract owner and name from the URL or name
                let owner = 'unknown'
                let repoName = repo.name

                // Try to extract from GitHub URL format
                const urlMatch = repo.url?.match(/github\.com\/([^\/]+)\/([^\/]+)/)
                if (urlMatch) {
                    [, owner, repoName] = urlMatch
                    repoName = repoName.replace('.git', '')
                }

                return {
                    id: `${owner}/${repoName}`,
                    name: repoName,
                    full_name: `${owner}/${repoName}`,
                    owner: { login: owner },
                    html_url: repo.url || `https://github.com/${owner}/${repoName}`,
                    clone_url: repo.url || `https://github.com/${owner}/${repoName}.git`,
                    description: `Analyzed ${repo.total_commits || 0} commits, ${repo.total_files || 0} files`,
                    private: false, // We don't have this info from the DiffSense API
                    stargazers_count: 0,
                    forks_count: 0,
                    language: repo.primary_language,
                    created_at: repo.created_at,
                    updated_at: repo.updated_at,
                    repo_id: repo.id, // Store the DiffSense repo ID (important!)
                    total_commits: repo.total_commits,
                    total_files: repo.total_files, status: repo.status,
                    source: 'diffsense', // Mark as from DiffSense API
                    isAnalyzed: true // Mark as already analyzed in DiffSense
                }
            })            // Replace repositories entirely with API data (no localStorage merge)
            setRepositories(formattedRepos)

        } catch (error) {
            console.error('Error loading analyzed repositories:', error)            // Fallback to localStorage only if API fails
            const savedRepos = localStorage.getItem('diffsense-repositories')
            if (savedRepos) {
                setRepositories(JSON.parse(savedRepos))
            }
        } finally {
            setIsLoadingRepos(false)
        }
    }

    const handleSelectRepo = (repo) => {
        selectRepository(repo)
        setIsMobileSidebarOpen(false) // Close mobile sidebar when repo is selected

        // Update recently viewed list
        setRecentlyViewed(prev => {
            const filtered = prev.filter(item => item.id !== repo.id)
            return [{ id: repo.id, timestamp: Date.now() }, ...filtered].slice(0, 50) // Keep only last 50
        })
    }

    const toggleFavorite = (repoId) => {
        setFavorites(prev =>
            prev.includes(repoId)
                ? prev.filter(id => id !== repoId) : [...prev, repoId]
        )
    }

    const handleAddRepository = async (owner, repoName, githubRepo = null) => {
        try {            // Construct the repository URL
            const repoUrl = `https://github.com/${owner}/${repoName}`

            // Check if user is authenticated and if this might be a private repo
            const useAuth = diffSenseAPI.isAuthenticated()

            // Use DiffSense API to clone and analyze the repository
            const result = await diffSenseAPI.cloneRepository(repoUrl, useAuth)
            // Create a repository object for the UI, preferring GitHub API data if available
            const newRepo = {
                id: `${owner}/${repoName}`,
                name: repoName,
                full_name: `${owner}/${repoName}`,
                owner: { login: owner },
                html_url: githubRepo?.clone_url?.replace('.git', '') || repoUrl,
                clone_url: githubRepo?.clone_url || `${repoUrl}.git`,
                description: githubRepo?.description || 'Repository added via URL',
                private: githubRepo?.private || false,
                stargazers_count: 0, // Not available from the API
                forks_count: 0, // Not available from the API
                language: githubRepo?.language || null,
                created_at: new Date().toISOString(),
                updated_at: new Date().toISOString(),
                repo_id: result.repo_id, // Store the DiffSense repo ID
                isAnalyzed: true // Mark as analyzed in DiffSense
            }            // Add to repositories list
            setRepositories(prev => {
                // Check if already exists
                if (prev.some(repo => repo.id === newRepo.id)) {
                    throw new Error('Repository already added')
                }
                return [...prev, newRepo]
            })

            // Immediately select the new repository with the existing repo_id
            // to avoid duplicate clone calls
            selectRepository(newRepo, result.repo_id)

            // Refresh the analyzed repositories list from the API
            // to get the most up-to-date information
            setTimeout(() => {
                loadAnalyzedRepositories()
            }, 1000) // Small delay to allow the backend to process

        } catch (error) {
            console.error('Error adding repository:', error)
            throw error
        }
    }

    const handleRemoveRepository = async (repoId) => {
        setRepositories(prev => prev.filter(repo => repo.id !== repoId))

        // Also remove from favorites and recently viewed
        setFavorites(prev => prev.filter(id => id !== repoId))
        setRecentlyViewed(prev => prev.filter(item => item.id !== repoId))

        // Clean up repository resources if we have the repo_id
        const repo = repositories.find(r => r.id === repoId)
        if (repo && repo.repo_id) {
            try {
                await diffSenseAPI.cleanupRepository(repo.repo_id)
            } catch (error) {
                console.warn('Failed to cleanup repository resources:', error)
            }
        }

        // Clear selection if removing selected repo
        if (selectedRepo?.id === repoId) {
            // Don't call setSelectedRepo as it doesn't exist in the context
            // The context will handle this automatically
        }
    }

    const filteredRepositories = repositories.filter(repo => {
        const matchesSearch = repo.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
            repo.full_name.toLowerCase().includes(searchQuery.toLowerCase())
        return matchesSearch
    })

    // Sort repositories by recently viewed, then by updated date
    const sortedRepositories = [...filteredRepositories].sort((a, b) => {
        const aRecentIndex = recentlyViewed.findIndex(item => item.id === a.id)
        const bRecentIndex = recentlyViewed.findIndex(item => item.id === b.id)

        // If both are in recently viewed, sort by recency
        if (aRecentIndex !== -1 && bRecentIndex !== -1) {
            return aRecentIndex - bRecentIndex
        }

        // If only one is in recently viewed, prioritize it
        if (aRecentIndex !== -1) return -1
        if (bRecentIndex !== -1) return 1        // If neither is in recently viewed, sort by updated date
        return new Date(b.updated_at) - new Date(a.updated_at)
    })

    return (
        <div className="h-screen bg-gray-50 dark:bg-gray-900 flex duration-200 transition-colors relative overflow-hidden">
            {/* Mobile sidebar overlay */}
            {isMobileSidebarOpen && (
                <div
                    className="fixed inset-0 bg-black bg-opacity-50 z-40 lg:hidden"
                    onClick={() => setIsMobileSidebarOpen(false)}
                />
            )}

            {/* Sidebar - Hidden on mobile by default, slides in when open */}
            <div className={`
                fixed lg:relative lg:translate-x-0 z-50 lg:z-auto
                transform transition-transform duration-300 ease-in-out
                ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}
                lg:flex lg:flex-shrink-0 h-full
            `}>                <Sidebar
                    repositories={sortedRepositories}
                    selectedRepo={selectedRepo}
                    onSelectRepo={handleSelectRepo}
                    onAddRepository={handleAddRepository}
                    onRemoveRepository={handleRemoveRepository}
                    favorites={favorites}
                    onToggleFavorite={toggleFavorite}
                    searchQuery={searchQuery}
                    onSearchChange={setSearchQuery}
                    user={user}
                    onLogout={onLogout}
                    setRepositories={setRepositories}
                    onCloseMobileSidebar={() => setIsMobileSidebarOpen(false)}
                    isRepositoryLoading={isRepositoryLoading}
                    onShowAddDialog={() => setShowAddDialog(true)}
                />
            </div>            {/* Main content area */}
            <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden">
                <MainContent user={user}
                    onToggleMobileSidebar={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
                    isMobileSidebarOpen={isMobileSidebarOpen}
                />
            </div>

            {/* Add Repository Dialog */}
            <AddRepositoryDialog
                isOpen={showAddDialog}
                onClose={() => setShowAddDialog(false)}
                onAddRepository={handleAddRepository}
                repositories={repositories}
            />
        </div>
    )
}

export default Dashboard
