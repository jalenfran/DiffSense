import { useState, useEffect } from 'react'
import {
    GitBranch,
    Star,
    GitFork,
    Clock,
    Code,
    ExternalLink,
    Activity,
    Calendar,
    Users,
    FileText,
    TrendingUp
} from 'lucide-react'
import axios from 'axios'

function MainContent({ selectedRepo, user }) {
    const [repoStats, setRepoStats] = useState(null)
    const [loading, setLoading] = useState(false)

    useEffect(() => {
        if (selectedRepo) {
            fetchRepoStats(selectedRepo)
        }
    }, [selectedRepo])
    const fetchRepoStats = async (repo) => {
        setLoading(true)
        try {
            // This would call your existing diffs endpoint
            const response = await axios.get(`/repos/${repo.owner.login}/${repo.name}/stats`)
            setRepoStats(response.data)
        } catch (error) {
            console.error('Failed to fetch repo stats:', error)
        } finally {
            setLoading(false)
        }
    }

    const formatDate = (dateString) => {
        return new Date(dateString).toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        })
    }

    const getLanguageColor = (language) => {
        const colors = {
            'JavaScript': 'bg-yellow-400',
            'TypeScript': 'bg-blue-400',
            'Python': 'bg-blue-500',
            'Java': 'bg-orange-500',
            'C++': 'bg-pink-500',
            'C#': 'bg-purple-500',
            'Go': 'bg-cyan-400',
            'Rust': 'bg-orange-600',
            'PHP': 'bg-indigo-500',
            'Ruby': 'bg-red-500',
        }
        return colors[language] || 'bg-gray-400'
    }

    if (!selectedRepo) {
        return (
            <div className="flex-1 flex items-center justify-center bg-gray-50 dark:bg-gray-900 duration-200 transition-colors">
                <div className="text-center">
                    <GitBranch className="w-16 h-16 text-gray-300 dark:text-gray-600 mx-auto mb-4" />
                    <h2 className="text-xl font-semibold text-gray-900 dark:text-white mb-2">Welcome to DiffSense</h2>
                    <p className="text-gray-600 dark:text-gray-400 max-w-md">
                        Select a repository from the sidebar to view detailed analysis and insights
                        about your code changes and development patterns.
                    </p>
                </div>
            </div>
        )
    }
    return (
        <div className="flex-1 overflow-y-auto bg-gray-50 dark:bg-gray-900 duration-200 transition-colors">
            <div className="p-6">
                {/* Repository Header */}
                <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mb-6">
                    <div className="flex items-start justify-between mb-4">
                        <div>
                            <h1 className="text-2xl font-bold text-gray-900 dark:text-white mb-2">
                                {selectedRepo.name}
                            </h1>
                            <p className="text-gray-600 dark:text-gray-300 mb-4">
                                {selectedRepo.description || 'No description available'}
                            </p>              <div className="flex items-center gap-4 text-sm text-gray-600 dark:text-gray-300">
                                <div className="flex items-center gap-1">
                                    <Users className="w-4 h-4" />
                                    <span>{selectedRepo.owner.login}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <Calendar className="w-4 h-4" />
                                    <span>Created {formatDate(selectedRepo.created_at)}</span>
                                </div>
                                <div className="flex items-center gap-1">
                                    <Clock className="w-4 h-4" />
                                    <span>Updated {formatDate(selectedRepo.updated_at)}</span>
                                </div>
                            </div>
                        </div>
                        <a
                            href={selectedRepo.html_url}
                            target="_blank"
                            rel="noopener noreferrer"
                            className="flex items-center gap-2 px-4 py-2 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 text-gray-700 dark:text-gray-300 rounded-lg transition-colors text-sm font-medium"
                        >
                            <ExternalLink className="w-4 h-4" />
                            View on GitHub
                        </a>
                    </div>

                    {/* Repository Stats */}
                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-1">
                                <Star className="w-4 h-4 text-yellow-500" />
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Stars</span>
                            </div>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {selectedRepo.stargazers_count.toLocaleString()}
                            </p>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-1">
                                <GitFork className="w-4 h-4 text-blue-500" />
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Forks</span>
                            </div>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {selectedRepo.forks_count.toLocaleString()}
                            </p>
                        </div>            <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-1">
                                <Code className="w-4 h-4 text-green-500" />
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Language</span>
                            </div>
                            <div className="flex items-center gap-2">
                                {selectedRepo.language && (
                                    <>
                                        <div className={`w-3 h-3 rounded-full ${getLanguageColor(selectedRepo.language)}`} />
                                        <span className="text-lg font-semibold text-gray-900 dark:text-white">
                                            {selectedRepo.language}
                                        </span>
                                    </>
                                )}
                            </div>
                        </div>

                        <div className="bg-gray-50 dark:bg-gray-700 rounded-lg p-4">
                            <div className="flex items-center gap-2 mb-1">
                                <FileText className="w-4 h-4 text-purple-500" />
                                <span className="text-sm font-medium text-gray-700 dark:text-gray-300">Size</span>
                            </div>
                            <p className="text-2xl font-bold text-gray-900 dark:text-white">
                                {(selectedRepo.size / 1024).toFixed(1)} MB
                            </p>
                        </div>
                    </div>
                </div>

                {/* Analysis Section */}
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">          {/* Repository Info */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <Activity className="w-5 h-5 text-primary-600" />
                            Repository Details
                        </h2>

                        <div className="space-y-4">
                            <div className="flex justify-between items-center py-2 border-b border-gray-100 dark:border-gray-700">
                                <span className="text-gray-600 dark:text-gray-400">Default Branch</span>
                                <span className="font-medium text-gray-900 dark:text-white">{selectedRepo.default_branch}</span>
                            </div>

                            <div className="flex justify-between items-center py-2 border-b border-gray-100 dark:border-gray-700">
                                <span className="text-gray-600 dark:text-gray-400">Visibility</span>
                                <span className={`px-2 py-1 rounded-full text-xs font-medium ${selectedRepo.private
                                    ? 'bg-red-100 dark:bg-red-900 text-red-800 dark:text-red-200'
                                    : 'bg-green-100 dark:bg-green-900 text-green-800 dark:text-green-200'
                                    }`}>
                                    {selectedRepo.private ? 'Private' : 'Public'}
                                </span>
                            </div>

                            <div className="flex justify-between items-center py-2 border-b border-gray-100 dark:border-gray-700">
                                <span className="text-gray-600 dark:text-gray-400">Open Issues</span>
                                <span className="font-medium text-gray-900 dark:text-white">{selectedRepo.open_issues_count}</span>
                            </div>

                            <div className="flex justify-between items-center py-2">
                                <span className="text-gray-600 dark:text-gray-400">Watchers</span>
                                <span className="font-medium text-gray-900 dark:text-white">{selectedRepo.watchers_count}</span>
                            </div>
                        </div>
                    </div>

                    {/* Diff Analysis Placeholder */}
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4 flex items-center gap-2">
                            <TrendingUp className="w-5 h-5 text-primary-600" />
                            Diff Analysis
                        </h2>

                        {loading ? (
                            <div className="flex items-center justify-center h-32">
                                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                            </div>
                        ) : (
                            <div className="text-center py-8">
                                <GitBranch className="w-12 h-12 text-gray-300 dark:text-gray-600 mx-auto mb-3" />
                                <p className="text-gray-600 dark:text-gray-400 mb-4">
                                    Diff analysis will be displayed here once implemented.
                                </p>
                                <button className="px-4 py-2 bg-primary-600 hover:bg-primary-700 text-white rounded-lg transition-colors font-medium">
                                    Analyze Repository
                                </button>
                            </div>
                        )}
                    </div>
                </div>        {/* Topics */}
                {selectedRepo.topics && selectedRepo.topics.length > 0 && (
                    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-6 mt-6">
                        <h2 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">Topics</h2>
                        <div className="flex flex-wrap gap-2">
                            {selectedRepo.topics.map((topic) => (
                                <span
                                    key={topic}
                                    className="px-3 py-1 bg-primary-100 dark:bg-primary-900 text-primary-800 dark:text-primary-200 rounded-full text-sm font-medium"
                                >
                                    {topic}
                                </span>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    )
}

export default MainContent
