import { useState, useEffect } from 'react'
import { Search, Star, Lock, Globe, Plus, LogOut, User, GitBranch, X, Loader2, Moon, Sun } from 'lucide-react'
import axios from 'axios'
import { useDarkMode } from '../contexts/DarkModeContext'

function Sidebar({
    repositories,
    selectedRepo,
    onSelectRepo,
    onAddRepository,
    onRemoveRepository,
    favorites,
    onToggleFavorite,
    searchQuery,
    onSearchChange,
    user,
    onLogout,
    setRepositories,
    onCloseMobileSidebar
}) {
    const [showFavoritesOnly, setShowFavoritesOnly] = useState(false)
    const [showAddDialog, setShowAddDialog] = useState(false)
    const [newRepoUrl, setNewRepoUrl] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [availableRepos, setAvailableRepos] = useState([])
    const [loadingRepos, setLoadingRepos] = useState(false)
    const [addMethod, setAddMethod] = useState('browse') // 'browse' or 'url'
    const [repoSearchQuery, setRepoSearchQuery] = useState('')

    const { darkMode, toggleDarkMode } = useDarkMode()
    const displayedRepos = showFavoritesOnly
        ? repositories.filter(repo => favorites.includes(repo.id))
        : repositories

    // Filter available repos for the dialog
    const filteredAvailableRepos = availableRepos.filter(repo => {
        const alreadyAdded = repositories.some(addedRepo => addedRepo.id === repo.id)
        const matchesSearch = repo.name.toLowerCase().includes(repoSearchQuery.toLowerCase()) ||
            repo.full_name.toLowerCase().includes(repoSearchQuery.toLowerCase())
        return !alreadyAdded && matchesSearch
    })

    // Fetch user's repositories when dialog opens
    useEffect(() => {
        if (showAddDialog && addMethod === 'browse' && availableRepos.length === 0) {
            fetchAvailableRepositories()
        }
    }, [showAddDialog, addMethod])

    const fetchAvailableRepositories = async () => {
        setLoadingRepos(true)
        try {
            const response = await axios.get('/repositories')
            setAvailableRepos(response.data)
        } catch (error) {
            console.error('Failed to fetch repositories:', error)
            alert('Failed to load your repositories')
        } finally {
            setLoadingRepos(false)
        }
    }

    const handleLogout = () => {
        window.location.href = 'http://localhost:3000/auth/logout'
        onLogout()
    }
    const handleAddRepository = async (repoData = null) => {
        setIsLoading(true)
        try {
            if (repoData) {
                // Adding from browse list - we already have the repo data
                if (repositories.some(repo => repo.id === repoData.id)) {
                    throw new Error('Repository already added')
                }
                setRepositories(prev => [...prev, repoData])
                setShowAddDialog(false)
                setRepoSearchQuery('')
            } else {
                // Adding from URL
                if (!newRepoUrl.trim()) return

                // Parse GitHub URL to extract owner/repo
                const match = newRepoUrl.match(/github\.com\/([^\/]+)\/([^\/]+)/)
                if (!match) {
                    alert('Please enter a valid GitHub repository URL')
                    return
                }

                const [, owner, repo] = match
                await onAddRepository(owner, repo.replace('.git', ''))
                setNewRepoUrl('')
                setShowAddDialog(false)
            }
        } catch (error) {
            console.error('Failed to add repository:', error)
            alert('Failed to add repository. Please check and try again.')
        } finally {
            setIsLoading(false)
        }
    }

    const getLanguageColor = (language) => {
        const colors = {
            'JavaScript': 'bg-yellow-400',
            'TypeScript': 'bg-blue-500',
            'Python': 'bg-green-500',
            'Java': 'bg-orange-500',
            'Go': 'bg-cyan-500',
            'Rust': 'bg-orange-600',
            'C++': 'bg-blue-600',
            'C#': 'bg-purple-600',
            'Ruby': 'bg-red-500',
            'PHP': 'bg-indigo-500',
            'Swift': 'bg-orange-400',
            'Kotlin': 'bg-purple-500',
            'Dart': 'bg-blue-400',
            'HTML': 'bg-orange-400',
            'CSS': 'bg-blue-400'
        }
        return colors[language] || 'bg-gray-400'
    }

    return (
        <div className="w-80 sm:w-72 lg:w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-screen flex flex-col duration-200 transition-colors">
            {/* Header Section */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700 duration-200 transition-colors">
                {/* Logo and Controls */}
                <div className="flex items-center justify-between mb-4">
                    <div className="flex items-center gap-3">
                        <div className="bg-primary-600 p-2 rounded-lg">
                            <GitBranch className="w-6 h-6 text-white" />
                        </div>
                        <h1 className="text-xl font-bold text-gray-900 dark:text-white">DiffSense</h1>
                    </div>
                    <div className="flex items-center gap-2">
                        <button
                            onClick={toggleDarkMode}
                            className="text-gray-400 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-200 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                            title={darkMode ? "Switch to light mode" : "Switch to dark mode"}
                        >
                            {darkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                        </button>
                        {/* Mobile close button */}
                        {onCloseMobileSidebar && (
                            <button
                                onClick={onCloseMobileSidebar}
                                className="lg:hidden text-gray-400 hover:text-gray-600 dark:text-gray-400 dark:hover:text-gray-200 p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                                title="Close sidebar"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        )}
                    </div>
                </div>                {/* User Info */}
                <div className="flex items-center gap-3 p-3 bg-gray-50 dark:bg-gray-700 rounded-lg mb-4">
                    <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full flex items-center justify-center">
                        {user.photos && user.photos[0] ? (
                            <img
                                src={user.photos[0].value}
                                alt={user.displayName || user.username}
                                className="w-8 h-8 rounded-full"
                            />
                        ) : (
                            <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                        )}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 dark:text-white text-sm truncate">
                            {user.displayName || user.username}
                        </p>
                        <p className="text-gray-500 dark:text-gray-400 text-xs truncate">@{user.username}</p>
                    </div>
                    <button
                        onClick={handleLogout}
                        className="text-gray-400 hover:text-gray-600 dark:text-gray-300 dark:hover:text-gray-100 p-1 rounded transition-colors"
                        title="Logout"
                    >
                        <LogOut className="w-4 h-4" />
                    </button>
                </div>
            </div>      {/* Repository Management */}
            <div className="p-4 border-b border-gray-200 dark:border-gray-700">
                {/* Search */}
                <div className="relative mb-3">
                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 w-4 h-4" />
                    <input
                        type="text"
                        placeholder="Search repositories..."
                        value={searchQuery}
                        onChange={(e) => onSearchChange(e.target.value)}
                        className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent text-sm bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                    />
                </div>

                {/* Add Repository Button */}
                <button
                    onClick={() => setShowAddDialog(true)}
                    className="w-full flex items-center gap-2 px-3 py-2 bg-primary-600 hover:bg-primary-700 dark:bg-primary-700 dark:hover:bg-primary-800 text-white rounded-lg text-sm font-medium transition-colors mb-3"
                >
                    <Plus className="w-4 h-4" />
                    Add Repository
                </button>

                {/* Favorites Toggle */}
                <button
                    onClick={() => setShowFavoritesOnly(!showFavoritesOnly)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors w-full ${showFavoritesOnly
                        ? 'bg-yellow-100 dark:bg-yellow-900 text-yellow-800 dark:text-yellow-200'
                        : 'text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                        }`}
                >
                    <Star className={`w-4 h-4 ${showFavoritesOnly ? 'fill-current' : ''}`} />
                    {showFavoritesOnly ? 'Show All' : 'Favorites Only'}
                </button>
            </div>

            {/* Repository List - Scrollable */}
            <div className="flex-1 overflow-y-auto">
                <div className="p-2">
                    {displayedRepos.length > 0 ? (
                        displayedRepos.map((repo) => (<div
                            key={repo.id}
                            onClick={() => onSelectRepo(repo)}
                            className={`group cursor-pointer rounded-lg p-3 mb-2 transition-all duration-200 ${selectedRepo?.id === repo.id
                                ? 'bg-primary-50 dark:bg-primary-900/30 border-primary-200 dark:border-primary-700 border'
                                : 'hover:bg-gray-50 dark:hover:bg-gray-700'
                                }`}
                        >
                            <div className="flex items-center justify-between">
                                <div className="flex-1 min-w-0">
                                    <div className="flex items-center gap-2">
                                        <h3 className="font-medium text-gray-900 dark:text-white truncate text-sm">
                                            {repo.name}
                                        </h3>
                                        {repo.private ? (
                                            <Lock className="w-3 h-3 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                        ) : (
                                            <Globe className="w-3 h-3 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                        )}
                                    </div>
                                </div>

                                <div className="flex items-center gap-1">
                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            onToggleFavorite(repo.id)
                                        }}
                                        className={`p-1 rounded transition-colors ${favorites.includes(repo.id)
                                            ? 'text-yellow-500 hover:text-yellow-600'
                                            : 'text-gray-300 dark:text-gray-600 hover:text-yellow-400'
                                            }`}
                                    >
                                        <Star className={`w-3 h-3 ${favorites.includes(repo.id) ? 'fill-current' : ''}`} />
                                    </button>

                                    <button
                                        onClick={(e) => {
                                            e.stopPropagation()
                                            if (confirm('Remove this repository from DiffSense?')) {
                                                onRemoveRepository(repo.id)
                                            }
                                        }}
                                        className="opacity-0 group-hover:opacity-100 p-1 rounded transition-all text-gray-400 dark:text-gray-500 hover:text-red-500 dark:hover:text-red-400"
                                    >
                                        <X className="w-3 h-3" />
                                    </button>
                                </div>
                            </div>
                        </div>
                        ))) : (
                        <div className="text-center py-8 text-gray-500 dark:text-gray-400">
                            <GitBranch className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                            <p className="text-sm mb-2">
                                {showFavoritesOnly ? 'No favorite repositories' : 'No repositories added yet'}
                            </p>
                            {!showFavoritesOnly && (
                                <p className="text-xs text-gray-400 dark:text-gray-500">
                                    Click "Add Repository" to get started
                                </p>
                            )}
                        </div>
                    )}
                </div>
            </div>      {/* Repository Count */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
                <p className="text-xs text-gray-600 dark:text-gray-400">
                    {displayedRepos.length} of {repositories.length} repositories
                </p>
            </div>      {/* Add Repository Dialog */}
            {showAddDialog && (
                <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-50 p-4" style={{ width: "100vw" }}>
                    <div className="bg-white dark:bg-gray-800 rounded-lg w-[600px] max-w-full h-[600px] max-h-[90vh] flex flex-col">
                        {/* Dialog Header */}
                        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
                            <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Add Repository</h2>
                            <button
                                onClick={() => {
                                    setShowAddDialog(false)
                                    setRepoSearchQuery('')
                                    setNewRepoUrl('')
                                }}
                                className="text-gray-400 dark:text-gray-500 hover:text-gray-600 dark:hover:text-gray-300"
                            >
                                <X className="w-5 h-5" />
                            </button>
                        </div>

                        {/* Method Toggle */}
                        <div className="px-6 pt-4 flex-shrink-0">
                            <div className="flex bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
                                <button
                                    onClick={() => setAddMethod('browse')}
                                    className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${addMethod === 'browse'
                                        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                                        : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                                        }`}
                                >
                                    Browse Your Repos
                                </button>
                                <button
                                    onClick={() => setAddMethod('url')}
                                    className={`flex-1 px-3 py-2 rounded-md text-sm font-medium transition-colors ${addMethod === 'url'
                                        ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                                        : 'text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white'
                                        }`}
                                >
                                    Add by URL
                                </button>
                            </div>
                        </div>

                        {/* Content */}
                        <div className="flex-1 min-h-0 p-6">
                            {addMethod === 'browse' ? (
                                <div className="h-full flex flex-col">                  {/* Search */}
                                    <div className="relative mb-4 flex-shrink-0">
                                        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 w-4 h-4" />
                                        <input
                                            type="text"
                                            placeholder="Search your repositories..."
                                            value={repoSearchQuery}
                                            onChange={(e) => setRepoSearchQuery(e.target.value)}
                                            className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                        />
                                    </div>

                                    {/* Repository List */}
                                    <div className="flex-1 min-h-0 overflow-y-auto border border-gray-200 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700">
                                        {loadingRepos ? (
                                            <div className="flex items-center justify-center p-8">
                                                <Loader2 className="w-6 h-6 animate-spin text-gray-400 dark:text-gray-500" />
                                                <span className="ml-2 text-gray-600 dark:text-gray-300">Loading repositories...</span>
                                            </div>
                                        ) : filteredAvailableRepos.length > 0 ? (
                                            <div className="p-2">
                                                {filteredAvailableRepos.map((repo) => (
                                                    <div
                                                        key={repo.id}
                                                        onClick={() => handleAddRepository(repo)}
                                                        className="group cursor-pointer rounded-lg p-3 mb-2 hover:bg-gray-50 dark:hover:bg-gray-600 border border-transparent hover:border-gray-200 dark:hover:border-gray-500 transition-all"
                                                    >
                                                        <div className="flex items-center justify-between">
                                                            <div className="flex-1 min-w-0">
                                                                <div className="flex items-center gap-2 mb-1">
                                                                    <h3 className="font-medium text-gray-900 dark:text-white truncate text-sm">
                                                                        {repo.name}
                                                                    </h3>
                                                                    {repo.private ? (
                                                                        <Lock className="w-3 h-3 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                                                    ) : (
                                                                        <Globe className="w-3 h-3 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                                                    )}
                                                                    {repo.language && (
                                                                        <div className="flex items-center gap-1">
                                                                            <div className={`w-2 h-2 rounded-full ${getLanguageColor(repo.language)}`}></div>
                                                                            <span className="text-xs text-gray-600 dark:text-gray-300">{repo.language}</span>
                                                                        </div>
                                                                    )}
                                                                </div>
                                                                {repo.description && (
                                                                    <p className="text-xs text-gray-500 dark:text-gray-400 truncate">{repo.description}</p>
                                                                )}
                                                                <div className="flex items-center gap-3 mt-1">
                                                                    <div className="flex items-center gap-1">
                                                                        <Star className="w-3 h-3 text-gray-400 dark:text-gray-500" />
                                                                        <span className="text-xs text-gray-500 dark:text-gray-400">{repo.stargazers_count}</span>
                                                                    </div>
                                                                    <div className="flex items-center gap-1">
                                                                        <GitBranch className="w-3 h-3 text-gray-400 dark:text-gray-500" />
                                                                        <span className="text-xs text-gray-500 dark:text-gray-400">{repo.forks_count}</span>
                                                                    </div>
                                                                    <span className="text-xs text-gray-500 dark:text-gray-400">
                                                                        Updated {new Date(repo.updated_at).toLocaleDateString()}
                                                                    </span>
                                                                </div>
                                                            </div>
                                                            <div className="ml-3 opacity-0 group-hover:opacity-100 transition-opacity">
                                                                <Plus className="w-4 h-4 text-primary-600 dark:text-primary-400" />
                                                            </div>
                                                        </div>
                                                    </div>
                                                ))}
                                            </div>
                                        ) : (
                                            <div className="flex items-center justify-center p-8 text-gray-500 dark:text-gray-400">
                                                <div className="text-center">
                                                    <GitBranch className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                                                    <p className="text-sm">
                                                        {repoSearchQuery ? 'No repositories match your search' : 'No available repositories'}
                                                    </p>
                                                    <p className="text-xs text-gray-400 dark:text-gray-500 mt-1">
                                                        All your repositories have been added
                                                    </p>
                                                </div>
                                            </div>
                                        )}
                                    </div>
                                </div>) : (
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                        GitHub Repository URL
                                    </label>
                                    <input
                                        type="text"
                                        placeholder="https://github.com/owner/repository"
                                        value={newRepoUrl}
                                        onChange={(e) => setNewRepoUrl(e.target.value)}
                                        className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent mb-2 bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                        onKeyPress={(e) => e.key === 'Enter' && handleAddRepository()}
                                    />
                                    <p className="text-xs text-gray-500 dark:text-gray-400">
                                        Enter the URL of any GitHub repository you have access to
                                    </p>
                                </div>
                            )}
                        </div>

                        {/* Dialog Footer */}
                        {addMethod === 'url' && (
                            <div className="flex gap-3 justify-end p-6 border-t border-gray-200 dark:border-gray-700 flex-shrink-0">
                                <button
                                    onClick={() => {
                                        setShowAddDialog(false)
                                        setNewRepoUrl('')
                                    }}
                                    className="px-4 py-2 text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                                >
                                    Cancel
                                </button>
                                <button
                                    onClick={() => handleAddRepository()}
                                    disabled={isLoading || !newRepoUrl.trim()}
                                    className="px-4 py-2 bg-primary-600 hover:bg-primary-700 dark:bg-primary-700 dark:hover:bg-primary-800 text-white rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
                                >
                                    {isLoading && <Loader2 className="w-4 h-4 animate-spin" />}
                                    {isLoading ? 'Adding...' : 'Add Repository'}
                                </button>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    )
}

export default Sidebar
