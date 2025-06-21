import { useState, useEffect } from 'react'
import Sidebar from './Sidebar'
import MainContent from './MainContent'
import axios from 'axios'

function Dashboard({ user, onLogout }) {
    const [repositories, setRepositories] = useState([])
    const [selectedRepo, setSelectedRepo] = useState(null)
    const [favorites, setFavorites] = useState(() => {
        const saved = localStorage.getItem('diffsense-favorites')
        return saved ? JSON.parse(saved) : []
    })
    const [recentlyViewed, setRecentlyViewed] = useState(() => {
        const saved = localStorage.getItem('diffsense-recently-viewed')
        return saved ? JSON.parse(saved) : []
    })
    const [searchQuery, setSearchQuery] = useState('')

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
        setSelectedRepo(repo)

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
        try {
            // Fetch repository details from GitHub API
            const response = await axios.get(`/repos/${owner}/${repoName}/stats`)
            const repoData = response.data.repository

            // Check if repo already exists
            if (repositories.some(repo => repo.id === repoData.id)) {
                throw new Error('Repository already added')
            }

            setRepositories(prev => [...prev, repoData])
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
        if (bRecentIndex !== -1) return 1
        // If neither is in recently viewed, sort by updated date
        return new Date(b.updated_at) - new Date(a.updated_at)
    })

    return (
        <div className="min-h-screen bg-gray-50 dark:bg-gray-900 flex duration-200 transition-colors">
            <Sidebar
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
            />

            <MainContent
                selectedRepo={selectedRepo} user={user}
            />
        </div>
    )
}

export default Dashboard
