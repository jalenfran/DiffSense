import React, { useState } from 'react'
import axios from 'axios'
import { Github, Loader2, CheckCircle, AlertCircle } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

const RepositoryCloner = ({ onRepositoryCloned }) => {
  const [repoUrl, setRepoUrl] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    
    if (!repoUrl.trim()) {
      setError('Please enter a repository URL')
      return
    }

    setLoading(true)
    setError('')

    try {
      const response = await axios.post(`${API_BASE}/api/clone-repository`, {
        repo_url: repoUrl.trim()
      })

      onRepositoryCloned(response.data)
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Failed to clone repository'
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <div className="text-center mb-6">
          <Github className="w-12 h-12 text-gray-400 mx-auto mb-4" />
          <h3 className="text-xl font-semibold text-gray-900 mb-2">
            Analyze Repository
          </h3>
          <p className="text-gray-600">
            Enter a GitHub repository URL to start analyzing semantic drift
          </p>
        </div>

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label htmlFor="repo-url" className="block text-sm font-medium text-gray-700 mb-2">
              Repository URL
            </label>
            <input
              id="repo-url"
              type="url"
              value={repoUrl}
              onChange={(e) => setRepoUrl(e.target.value)}
              placeholder="https://github.com/username/repository"
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
              disabled={loading}
            />
          </div>

          {error && (
            <div className="flex items-center space-x-2 text-red-600 text-sm">
              <AlertCircle className="w-4 h-4" />
              <span>{error}</span>
            </div>
          )}

          <button
            type="submit"
            disabled={loading || !repoUrl.trim()}
            className="w-full btn-primary flex items-center justify-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                <span>Cloning Repository...</span>
              </>
            ) : (
              <>
                <Github className="w-4 h-4" />
                <span>Clone & Analyze</span>
              </>
            )}
          </button>
        </form>

        <div className="mt-6 pt-6 border-t border-gray-200">
          <h4 className="text-sm font-medium text-gray-900 mb-3">Example repositories:</h4>
          <div className="space-y-2">
            {[
              'https://github.com/microsoft/vscode',
              'https://github.com/facebook/react',
              'https://github.com/torvalds/linux'
            ].map((url) => (
              <button
                key={url}
                onClick={() => setRepoUrl(url)}
                className="block w-full text-left text-sm text-primary-600 hover:text-primary-700 hover:bg-primary-50 px-2 py-1 rounded transition-colors"
                disabled={loading}
              >
                {url}
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}

export default RepositoryCloner
