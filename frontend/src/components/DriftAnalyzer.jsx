import React, { useState, useEffect } from 'react'
import axios from 'axios'
import FileSelector from './FileSelector'
import DriftTimeline from './DriftTimeline'
import DriftSummary from './DriftSummary'
import { BarChart3, FileText, Code, Loader2 } from 'lucide-react'

const API_BASE = 'http://localhost:8000'

const DriftAnalyzer = ({ repoData }) => {
  const [selectedFile, setSelectedFile] = useState('')
  const [functionName, setFunctionName] = useState('')
  const [analysisType, setAnalysisType] = useState('file') // 'file' or 'function'
  const [analysis, setAnalysis] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a file to analyze')
      return
    }

    if (analysisType === 'function' && !functionName) {
      setError('Please enter a function name')
      return
    }

    setLoading(true)
    setError('')
    setAnalysis(null)

    try {
      const endpoint = analysisType === 'file' 
        ? `/api/analyze-file/${repoData.repo_id}`
        : `/api/analyze-function/${repoData.repo_id}`

      const payload = analysisType === 'file'
        ? {
            repo_path: '',
            file_path: selectedFile,
            max_commits: 50
          }
        : {
            repo_path: '',
            file_path: selectedFile,
            function_name: functionName,
            max_commits: 30
          }

      const response = await axios.post(`${API_BASE}${endpoint}`, payload)
      setAnalysis(response.data)
    } catch (err) {
      const errorMessage = err.response?.data?.detail || 'Analysis failed'
      setError(errorMessage)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Repository Info */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-xl font-semibold text-gray-900 mb-2">
              Repository Analysis
            </h2>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-500">Total Commits:</span>
                <span className="ml-2 font-medium">{repoData.stats.total_commits}</span>
              </div>
              <div>
                <span className="text-gray-500">Contributors:</span>
                <span className="ml-2 font-medium">{repoData.stats.contributors}</span>
              </div>
              <div>
                <span className="text-gray-500">Branches:</span>
                <span className="ml-2 font-medium">{repoData.stats.active_branches}</span>
              </div>
              <div>
                <span className="text-gray-500">Files:</span>
                <span className="ml-2 font-medium">{repoData.stats.file_count}</span>
              </div>
            </div>
          </div>
          <BarChart3 className="w-8 h-8 text-primary-500" />
        </div>
      </div>

      {/* Analysis Configuration */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Configure Analysis</h3>
        
        {/* Analysis Type Selector */}
        <div className="mb-4">
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Analysis Type
          </label>
          <div className="grid grid-cols-2 gap-3">
            <button
              onClick={() => setAnalysisType('file')}
              className={`p-3 rounded-lg border-2 transition-colors ${
                analysisType === 'file'
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <FileText className="w-5 h-5 mx-auto mb-1" />
              <span className="text-sm font-medium">Analyze File</span>
            </button>
            <button
              onClick={() => setAnalysisType('function')}
              className={`p-3 rounded-lg border-2 transition-colors ${
                analysisType === 'function'
                  ? 'border-primary-500 bg-primary-50 text-primary-700'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
            >
              <Code className="w-5 h-5 mx-auto mb-1" />
              <span className="text-sm font-medium">Analyze Function</span>
            </button>
          </div>
        </div>

        {/* File Selector */}
        <FileSelector
          repoId={repoData.repo_id}
          selectedFile={selectedFile}
          onFileSelect={setSelectedFile}
        />

        {/* Function Name Input (if function analysis) */}
        {analysisType === 'function' && (
          <div className="mt-4">
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Function Name
            </label>
            <input
              type="text"
              value={functionName}
              onChange={(e) => setFunctionName(e.target.value)}
              placeholder="e.g., calculateTotal, handleClick, etc."
              className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-primary-500 focus:border-primary-500"
            />
          </div>
        )}

        {/* Error Display */}
        {error && (
          <div className="mt-4 p-3 bg-red-50 border border-red-200 rounded-md">
            <p className="text-sm text-red-600">{error}</p>
          </div>
        )}

        {/* Analyze Button */}
        <button
          onClick={handleAnalyze}
          disabled={loading || !selectedFile || (analysisType === 'function' && !functionName)}
          className="mt-4 btn-primary w-full disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              <span>Analyzing...</span>
            </>
          ) : (
            <>
              <BarChart3 className="w-4 h-4" />
              <span>Start Analysis</span>
            </>
          )}
        </button>
      </div>

      {/* Analysis Results */}
      {analysis && (
        <div className="space-y-6">
          <DriftSummary analysis={analysis} />
          <DriftTimeline analysis={analysis} />
        </div>
      )}
    </div>
  )
}

export default DriftAnalyzer
