import { useState, useEffect } from 'react'
import { Search, Star, Lock, Globe, Plus, X, Loader2 } from 'lucide-react'
import axios from 'axios'
import { diffSenseAPI } from '../services/api'

function AddRepositoryDialog({ 
    isOpen, 
    onClose, 
    onAddRepository,
    repositories 
}) {
    const [newRepoUrl, setNewRepoUrl] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [availableRepos, setAvailableRepos] = useState([])
    const [loadingRepos, setLoadingRepos] = useState(false)
    const [addMethod, setAddMethod] = useState('url') // 'browse' or 'url'
    const [repoSearchQuery, setRepoSearchQuery] = useState('')

    // Filter available repos for the dialog
    const filteredAvailableRepos = availableRepos.filter(repo => {
        const alreadyAdded = repositories.some(addedRepo => addedRepo.id === repo.id)
        const matchesSearch = repo.name.toLowerCase().includes(repoSearchQuery.toLowerCase()) ||
            repo.full_name.toLowerCase().includes(repoSearchQuery.toLowerCase())
        return !alreadyAdded && matchesSearch
    })

    // Fetch user's repositories when dialog opens
    useEffect(() => {
        if (isOpen && addMethod === 'browse' && availableRepos.length === 0) {
            fetchAvailableRepositories()
        }
    }, [isOpen, addMethod])

    const fetchAvailableRepositories = async () => {
        setLoadingRepos(true)
        try {
            const response = await axios.get('/auth/repositories', {
                headers: {
                    'Content-Type': 'application/json'
                }
            })
            
            if (response.data && Array.isArray(response.data)) {
                setAvailableRepos(response.data)
            } else {
                console.error('Invalid repositories response:', response.data)
                // Fallback - set empty array so user can still add by URL
                setAvailableRepos([])
            }
        } catch (error) {
            console.error('Error fetching repositories:', error)
            // Fallback - set empty array so user can still add by URL
            setAvailableRepos([])
        } finally {
            setLoadingRepos(false)        }
    }

    const handleAddRepository = async (repoData = null) => {
        if (isLoading) return

        setIsLoading(true)

        try {
            let repoUrl
            let owner, repoName

            if (repoData) {
                // Adding from browse list
                owner = repoData.owner.login
                repoName = repoData.name
                repoUrl = repoData.clone_url || repoData.html_url
            } else {
                // Adding from URL
                if (!newRepoUrl.trim()) {
                    throw new Error('Please enter a repository URL')
                }

                const urlPattern = /github\.com\/([^\/]+)\/([^\/]+)/
                const match = newRepoUrl.match(urlPattern)
                
                if (!match) {
                    throw new Error('Please enter a valid GitHub repository URL')
                }

                [, owner, repoName] = match
                repoName = repoName.replace('.git', '')
                repoUrl = newRepoUrl
            }

            // Call the actual API to add the repository
            await onAddRepository(owner, repoName)

            // Close dialog and reset form
            onClose()
            setRepoSearchQuery('')
            setNewRepoUrl('')
        } catch (error) {
            console.error('Error adding repository:', error)
            alert(error.message || 'Failed to add repository')
        } finally {
            setIsLoading(false)
        }
    }

    const handleClose = () => {
        onClose()
        setRepoSearchQuery('')
        setNewRepoUrl('')
    }

    if (!isOpen) return null

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-[9999] p-4" style={{ width: "100vw" }}>
            <div className="bg-white dark:bg-gray-800 rounded-lg w-[600px] max-w-full h-[600px] max-h-[90vh] flex flex-col">
                {/* Dialog Header */}
                <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700 flex-shrink-0">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white">Add Repository</h2>
                    <button
                        onClick={handleClose}
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

                {/* Content Area */}
                <div className="flex-1 overflow-hidden flex flex-col">
                    {addMethod === 'browse' ? (
                        <>
                            {/* Search */}
                            <div className="px-6 py-4 flex-shrink-0">
                                <div className="relative">
                                    <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 dark:text-gray-500 w-4 h-4" />
                                    <input
                                        type="text"
                                        placeholder="Search your repositories..."
                                        value={repoSearchQuery}
                                        onChange={(e) => setRepoSearchQuery(e.target.value)}
                                        className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                    />
                                </div>
                            </div>                            {/* Repository List */}
                            <div className="flex-1 overflow-y-auto px-6 pb-6">
                                {loadingRepos ? (
                                    <div className="flex items-center justify-center py-8">
                                        <Loader2 className="w-6 h-6 animate-spin text-gray-400 dark:text-gray-500" />
                                        <span className="ml-2 text-gray-600 dark:text-gray-400">Loading repositories...</span>
                                    </div>
                                ) : filteredAvailableRepos.length > 0 ? (
                                    <div className="space-y-2">
                                        {filteredAvailableRepos.map((repo) => (
                                            <div
                                                key={repo.id}
                                                className="flex items-center justify-between p-3 border border-gray-200 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                                            >
                                                <div className="flex-1 min-w-0">
                                                    <div className="flex items-center gap-2">
                                                        {repo.private ? (
                                                            <Lock className="w-4 h-4 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                                        ) : (
                                                            <Globe className="w-4 h-4 text-gray-400 dark:text-gray-500 flex-shrink-0" />
                                                        )}
                                                        <span className="font-medium text-gray-900 dark:text-white truncate">
                                                            {repo.name}
                                                        </span>
                                                        {repo.stargazers_count > 0 && (
                                                            <div className="flex items-center gap-1 text-yellow-500">
                                                                <Star className="w-3 h-3 fill-current" />
                                                                <span className="text-xs">{repo.stargazers_count}</span>
                                                            </div>
                                                        )}
                                                    </div>
                                                    {repo.description && (
                                                        <p className="text-sm text-gray-600 dark:text-gray-400 truncate mt-1">
                                                            {repo.description}
                                                        </p>
                                                    )}
                                                    <div className="flex items-center gap-4 mt-2 text-xs text-gray-500 dark:text-gray-400">
                                                        {repo.language && (
                                                            <span className="flex items-center gap-1">
                                                                <div className="w-2 h-2 rounded-full bg-blue-400"></div>
                                                                {repo.language}
                                                            </span>
                                                        )}
                                                        <span>Updated {new Date(repo.updated_at).toLocaleDateString()}</span>
                                                    </div>
                                                </div>
                                                <button
                                                    onClick={() => handleAddRepository(repo)}
                                                    disabled={isLoading}
                                                    className="ml-3 px-3 py-1 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white text-sm rounded-lg transition-colors flex items-center gap-1"
                                                >
                                                    <Plus className="w-3 h-3" />
                                                    Add
                                                </button>
                                            </div>
                                        ))}
                                    </div>
                                ) : (
                                    <div className="text-center py-8">
                                        <div className="text-gray-500 dark:text-gray-400">
                                            <p className="mb-2">Browse functionality is currently unavailable.</p>
                                            <p className="text-sm">Please use "Add by URL" to add repositories.</p>
                                        </div>
                                    </div>
                                )}
                            </div>
                        </>
                    ) : (
                        <div className="px-6 py-4 flex-1 flex flex-col">
                            <div className="mb-4">
                                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                                    Repository URL
                                </label>
                                <input
                                    type="url"
                                    placeholder="https://github.com/owner/repository"
                                    value={newRepoUrl}
                                    onChange={(e) => setNewRepoUrl(e.target.value)}
                                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
                                />
                            </div>
                            <div className="text-sm text-gray-600 dark:text-gray-400 mb-6">
                                Enter the full GitHub repository URL. The repository will be cloned and analyzed for insights.
                            </div>
                            <button
                                onClick={() => handleAddRepository()}
                                disabled={!newRepoUrl.trim() || isLoading}
                                className="w-full px-4 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-blue-400 text-white rounded-lg transition-colors flex items-center justify-center gap-2"
                            >
                                {isLoading ? (
                                    <>
                                        <Loader2 className="w-4 h-4 animate-spin" />
                                        Adding Repository...
                                    </>
                                ) : (
                                    <>
                                        <Plus className="w-4 h-4" />
                                        Add Repository
                                    </>
                                )}
                            </button>
                        </div>
                    )}
                </div>
            </div>
        </div>
    )
}

export default AddRepositoryDialog
