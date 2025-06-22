import { useState, useEffect } from 'react'
import { Search, Star, Lock, Globe, Plus, LogOut, User, GitBranch, X, Loader2, Moon, Sun } from 'lucide-react'
import { useDarkMode } from '../contexts/DarkModeContext'
import { diffSenseAPI } from '../services/api'

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
    onCloseMobileSidebar,
    isRepositoryLoading,
    onShowAddDialog
}) {
    const [showFavoritesOnly, setShowFavoritesOnly] = useState(false)

    const { darkMode, toggleDarkMode } = useDarkMode()
    const displayedRepos = showFavoritesOnly
        ? repositories.filter(repo => favorites.includes(repo.id))
        : repositories

    const handleLogout = async () => {
        try {
            await diffSenseAPI.logout()
        } catch (error) {
            console.error('Logout error:', error)
        } finally {
            onLogout()
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
        <div className="w-80 sm:w-72 lg:w-80 bg-white dark:bg-gray-800 border-r border-gray-200 dark:border-gray-700 h-full flex flex-col duration-200 transition-colors">
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
                        {user.github_avatar ? (
                            <img
                                src={user.github_avatar}
                                alt={user.display_name || user.github_username}
                                className="w-8 h-8 rounded-full"
                            />
                        ) : (
                            <User className="w-4 h-4 text-gray-600 dark:text-gray-300" />
                        )}
                    </div>
                    <div className="flex-1 min-w-0">
                        <p className="font-medium text-gray-900 dark:text-white text-sm truncate">
                            {user.display_name || user.github_username}
                        </p>
                        <p className="text-gray-500 dark:text-gray-400 text-xs truncate">@{user.github_username}</p>
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
                </div>                {/* Add Repository Button */}
                <button
                    onClick={onShowAddDialog}
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
                                <div className="flex-1 min-w-0">                                    <div className="flex items-center gap-2">
                                    <h3 className="font-medium text-gray-900 dark:text-white truncate text-sm">
                                        {repo.name}
                                    </h3>
                                    {selectedRepo?.id === repo.id && isRepositoryLoading && (
                                        <div className="animate-spin rounded-full h-3 w-3 border-b-2 border-blue-500"></div>
                                    )}
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
                </p>            </div>
        </div>
    )

}

export default Sidebar
