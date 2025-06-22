import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import MainContent from './MainContent'
import { useRepository } from '../contexts/RepositoryContext'

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

    const { selectRepository, selectedRepo, isLoading: isRepositoryLoading } = useRepository()

    useEffect(() => {
        localStorage.setItem('diffsense-favorites', JSON.stringify(favorites))
    }, [favorites])

    useEffect(() => {
        localStorage.setItem('diffsense-recently-viewed', JSON.stringify(recentlyViewed))
    }, [recentlyViewed])

    useEffect(() => {
        localStorage.setItem('diffsense-repositories', JSON.stringify(repositories))
    }, [repositories])

    // Load saved repositories from localStorage
    useEffect(() => {
        const savedRepos = localStorage.getItem('diffsense-repositories')
        if (savedRepos) {
            setRepositories(JSON.parse(savedRepos))
        }
    }, [])

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
                ? prev.filter(id => id !== repoId)
                : [...prev, repoId]
        )
    }

    const handleAddRepository = async (owner, repoName) => {
        // This is kept for compatibility with the browse functionality
        // The URL functionality is now handled in the Sidebar component
        try {
            // Fetch repository details from GitHub API (if needed for browse mode)
            // For now, this is just a placeholder
            console.log('Adding repository:', owner, repoName)
        } catch (error) {
            console.error('Error adding repository:', error)
            throw error
        }
    }

    const handleRemoveRepository = (repoId) => {
        setRepositories(prev => prev.filter(repo => repo.id !== repoId))

        // Also remove from favorites and recently viewed
        setFavorites(prev => prev.filter(id => id !== repoId))
        setRecentlyViewed(prev => prev.filter(item => item.id !== repoId))

        // Clear selection if removing selected repo
        if (selectedRepo?.id === repoId) {
            setSelectedRepo(null)
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
            `}><Sidebar
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
                />
            </div>            {/* Main content area */}
            <div className="flex-1 flex flex-col min-w-0 h-full overflow-hidden"><MainContent
                    user={user}
                    onToggleMobileSidebar={() => setIsMobileSidebarOpen(!isMobileSidebarOpen)}
                    isMobileSidebarOpen={isMobileSidebarOpen}
                />
            </div>
        </div>
    )
}

export default Dashboard
