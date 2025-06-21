import React, { useState } from 'react'
import RepositoryCloner from './components/RepositoryCloner'
import DriftAnalyzer from './components/DriftAnalyzer'
import { Layers, TrendingUp, GitBranch } from 'lucide-react'

function App() {
  const [repoData, setRepoData] = useState(null)

  const handleRepositoryCloned = (data) => {
    setRepoData(data)
  }

  const handleReset = () => {
    setRepoData(null)
  }

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Header */}
      <header className="bg-white shadow-sm border-b">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center py-6">
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-primary-600 rounded-lg">
                <TrendingUp className="w-6 h-6 text-white" />
              </div>
              <div>
                <h1 className="text-2xl font-bold text-gray-900">DiffSense</h1>
                <p className="text-sm text-gray-500">Feature Drift Detector</p>
              </div>
            </div>
            
            {repoData && (
              <button
                onClick={handleReset}
                className="btn-secondary flex items-center space-x-2"
              >
                <GitBranch className="w-4 h-4" />
                <span>New Repository</span>
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {!repoData ? (
          <div className="text-center">
            <div className="mb-8">
              <Layers className="w-16 h-16 text-primary-500 mx-auto mb-4" />
              <h2 className="text-3xl font-bold text-gray-900 mb-2">
                Detect Semantic Drift in Your Code
              </h2>
              <p className="text-lg text-gray-600 max-w-2xl mx-auto">
                Analyze how your features evolve over time using AI-powered semantic analysis 
                of git commits, diffs, and changes.
              </p>
            </div>
            
            <div className="grid md:grid-cols-3 gap-6 mb-12 max-w-4xl mx-auto">
              <div className="card text-center">
                <div className="w-12 h-12 bg-blue-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <TrendingUp className="w-6 h-6 text-blue-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Semantic Analysis</h3>
                <p className="text-sm text-gray-600">
                  Uses embeddings to understand the meaning behind code changes
                </p>
              </div>
              
              <div className="card text-center">
                <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <GitBranch className="w-6 h-6 text-green-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Git Integration</h3>
                <p className="text-sm text-gray-600">
                  Analyzes your entire git history to track feature evolution
                </p>
              </div>
              
              <div className="card text-center">
                <div className="w-12 h-12 bg-purple-100 rounded-lg flex items-center justify-center mx-auto mb-4">
                  <Layers className="w-6 h-6 text-purple-600" />
                </div>
                <h3 className="font-semibold text-gray-900 mb-2">Breaking Changes</h3>
                <p className="text-sm text-gray-600">
                  Predicts potentially risky changes before they impact users
                </p>
              </div>
            </div>
            
            <RepositoryCloner onRepositoryCloned={handleRepositoryCloned} />
          </div>
        ) : (
          <DriftAnalyzer repoData={repoData} />
        )}
      </main>
    </div>
  )
}

export default App
