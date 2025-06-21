import React, { useState, useEffect } from 'react'
import axios from 'axios'
import { Folder, File, Loader2, Search } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

const FileSelector = ({ repoId, selectedFile, onFileSelect }) => {
  const [files, setFiles] = useState([])
  const [loading, setLoading] = useState(false)
  const [searchTerm, setSearchTerm] = useState('')
  const [error, setError] = useState('')

  useEffect(() => {
    if (repoId) {
      fetchFiles()
    }
  }, [repoId])

  const fetchFiles = async () => {
    setLoading(true)
    setError('')

    try {
      const response = await axios.get(`${API_BASE}/api/repository/${repoId}/files`)
      setFiles(response.data.files)
    } catch (err) {
      setError('Failed to load files')
    } finally {
      setLoading(false)
    }
  }

  const filteredFiles = files.filter(file =>
    file.path.toLowerCase().includes(searchTerm.toLowerCase())
  )

  const getFileIcon = (filename) => {
    const ext = filename.split('.').pop()
    const iconMap = {
      'js': '📄',
      'jsx': '⚛️',
      'ts': '📘',
      'tsx': '⚛️',
      'py': '🐍',
      'java': '☕',
      'cpp': '⚙️',
      'c': '⚙️',
      'go': '🔧',
      'rs': '🦀',
      'rb': '💎',
      'php': '🐘'
    }
    return iconMap[ext] || '📄'
  }

  if (loading) {
    return (
      <div className="flex items-center justify-center p-8">
        <Loader2 className="w-6 h-6 animate-spin text-primary-500" />
        <span className="ml-2 text-gray-600">Loading files...</span>
      </div>
    )
  }

  if (error) {
    return (
      <div className="p-4 bg-red-50 border border-red-200 rounded-md">
        <p className="text-sm text-red-600">{error}</p>
      </div>
    )
  }

  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-2">
        Select File to Analyze
      </label>
      
      {/* Search Box */}
      <div className="relative mb-3">
        <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search files..."
          className="w-full pl-10 pr-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
        />
      </div>

      {/* File List */}
      <div className="max-h-64 overflow-y-auto border border-gray-200 rounded-md">
        {filteredFiles.length > 0 ? (
          <div className="divide-y divide-gray-200">
            {filteredFiles.map((file) => (
              <button
                key={file.path}
                onClick={() => onFileSelect(file.path)}
                className={`w-full text-left p-3 hover:bg-gray-50 transition-colors ${
                  selectedFile === file.path ? 'bg-primary-50 border-r-2 border-primary-500' : ''
                }`}
              >
                <div className="flex items-center space-x-3">
                  <span className="text-lg">{getFileIcon(file.path)}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {file.path.split('/').pop()}
                    </p>
                    <p className="text-xs text-gray-500 truncate">
                      {file.path}
                    </p>
                  </div>
                  <div className="text-xs text-gray-400">
                    {(file.size / 1024).toFixed(1)}KB
                  </div>
                </div>
              </button>
            ))}
          </div>
        ) : (
          <div className="p-8 text-center text-gray-500">
            {searchTerm ? (
              <>
                <Search className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p>No files found matching "{searchTerm}"</p>
              </>
            ) : (
              <>
                <File className="w-8 h-8 mx-auto mb-2 text-gray-400" />
                <p>No code files found in repository</p>
              </>
            )}
          </div>
        )}
      </div>

      {selectedFile && (
        <div className="mt-3 p-3 bg-primary-50 border border-primary-200 rounded-md">
          <p className="text-sm">
            <span className="font-medium text-primary-700">Selected:</span>
            <span className="ml-2 text-primary-600">{selectedFile}</span>
          </p>
        </div>
      )}
    </div>
  )
}

export default FileSelector
