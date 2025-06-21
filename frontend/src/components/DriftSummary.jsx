import React from 'react'
import { AlertTriangle, TrendingUp, GitCommit, Activity } from 'lucide-react'

const DriftSummary = ({ analysis }) => {
  const getDriftLevel = (score) => {
    if (score > 0.7) return { level: 'high', color: 'red', label: 'High' }
    if (score > 0.3) return { level: 'medium', color: 'yellow', label: 'Medium' }
    return { level: 'low', color: 'green', label: 'Low' }
  }

  const getRiskLevel = (riskLevel) => {
    const levels = {
      high: { color: 'red', icon: AlertTriangle },
      medium: { color: 'yellow', icon: AlertTriangle },
      low: { color: 'green', icon: Activity }
    }
    return levels[riskLevel] || levels.low
  }

  const overallDrift = getDriftLevel(analysis.overall_drift)
  const riskInfo = getRiskLevel(analysis.risk_assessment.risk_level)
  const RiskIcon = riskInfo.icon

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Analysis Summary</h3>
        <TrendingUp className="w-6 h-6 text-primary-500" />
      </div>

      {/* Key Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4 mb-6">
        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {analysis.overall_drift.toFixed(2)}
          </div>
          <div className="text-sm text-gray-600">Overall Drift Score</div>
          <div className={`mt-1 inline-block px-2 py-1 text-xs font-medium rounded drift-badge-${overallDrift.level}`}>
            {overallDrift.label}
          </div>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {analysis.total_commits}
          </div>
          <div className="text-sm text-gray-600">Total Commits</div>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {analysis.drift_events.length}
          </div>
          <div className="text-sm text-gray-600">Significant Changes</div>
        </div>

        <div className="text-center p-4 bg-gray-50 rounded-lg">
          <div className="text-2xl font-bold text-gray-900">
            {(analysis.risk_assessment.risk_score * 100).toFixed(0)}%
          </div>
          <div className="text-sm text-gray-600">Breaking Risk</div>
          <div className={`mt-1 inline-flex items-center space-x-1 px-2 py-1 text-xs font-medium rounded drift-badge-${riskInfo.color === 'red' ? 'high' : riskInfo.color === 'yellow' ? 'medium' : 'low'}`}>
            <RiskIcon className="w-3 h-3" />
            <span>{analysis.risk_assessment.risk_level.toUpperCase()}</span>
          </div>
        </div>
      </div>

      {/* Feature Info */}
      <div className="border-t pt-4">
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
          <div>
            <span className="font-medium text-gray-700">Target:</span>
            <span className="ml-2 text-gray-600">
              {analysis.function_name 
                ? `Function: ${analysis.function_name}` 
                : `File: ${analysis.file_path.split('/').pop()}`
              }
            </span>
          </div>
          <div>
            <span className="font-medium text-gray-700">File Path:</span>
            <span className="ml-2 text-gray-600 font-mono text-xs">
              {analysis.file_path}
            </span>
          </div>
        </div>
      </div>

      {/* Change Summary */}
      {analysis.change_summary && (
        <div className="border-t pt-4 mt-4">
          <h4 className="font-medium text-gray-900 mb-3">Change Patterns</h4>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-sm">
            <div>
              <span className="text-gray-500">Average Similarity:</span>
              <span className="ml-2 font-medium">
                {(analysis.change_summary.average_similarity || 0).toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Max Drift:</span>
              <span className="ml-2 font-medium">
                {(analysis.change_summary.max_drift || 0).toFixed(2)}
              </span>
            </div>
            <div>
              <span className="text-gray-500">Trend:</span>
              <span className="ml-2 font-medium capitalize">
                {analysis.change_summary.drift_trend || 'stable'}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Risk Assessment Details */}
      <div className="border-t pt-4 mt-4">
        <h4 className="font-medium text-gray-900 mb-2">Risk Assessment</h4>
        <p className="text-sm text-gray-600">
          {analysis.risk_assessment.reasoning}
        </p>
        {analysis.risk_assessment.recent_events_count > 0 && (
          <p className="text-xs text-gray-500 mt-1">
            Based on {analysis.risk_assessment.recent_events_count} recent significant changes
          </p>
        )}
      </div>
    </div>
  )
}

export default DriftSummary
