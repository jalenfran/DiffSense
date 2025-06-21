import React, { useState } from 'react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts'
import { Calendar, GitCommit, User, AlertCircle, TrendingUp } from 'lucide-react'

const DriftTimeline = ({ analysis }) => {
  const [selectedCommit, setSelectedCommit] = useState(null)

  // Prepare data for the chart
  const timelineData = analysis.timeline.map((point, index) => ({
    index: index + 1,
    ...point,
    driftScore: point.drift_score,
    date: new Date(point.timestamp).toLocaleDateString(),
    shortHash: point.commit_hash.substring(0, 7)
  }))

  const CustomTooltip = ({ active, payload, label }) => {
    if (active && payload && payload.length) {
      const data = payload[0].payload
      return (
        <div className="bg-white p-3 border border-gray-200 rounded-lg shadow-lg">
          <p className="font-medium text-gray-900">{`Commit ${data.shortHash}`}</p>
          <p className="text-sm text-gray-600">{data.date}</p>
          <p className="text-sm">
            <span className="font-medium">Drift Score:</span> {data.driftScore.toFixed(3)}
          </p>
          <p className="text-sm text-gray-600 mt-1 max-w-xs">
            {data.commit_message.substring(0, 80)}...
          </p>
        </div>
      )
    }
    return null
  }

  const handleCommitClick = (commit) => {
    setSelectedCommit(selectedCommit?.commit_hash === commit.commit_hash ? null : commit)
  }

  return (
    <div className="card">
      <div className="flex items-center justify-between mb-6">
        <h3 className="text-lg font-semibold text-gray-900">Semantic Drift Timeline</h3>
        <TrendingUp className="w-6 h-6 text-primary-500" />
      </div>

      {/* Chart */}
      <div className="mb-6">
        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={timelineData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
              <XAxis 
                dataKey="index" 
                tick={{ fontSize: 12 }}
                label={{ value: 'Commit Number', position: 'insideBottom', offset: -5 }}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                label={{ value: 'Drift Score', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip content={<CustomTooltip />} />
              <ReferenceLine y={0.3} stroke="#f59e0b" strokeDasharray="5 5" label="Significant Threshold" />
              <Line 
                type="monotone" 
                dataKey="driftScore" 
                stroke="#3b82f6" 
                strokeWidth={2}
                dot={{ r: 4, fill: '#3b82f6' }}
                activeDot={{ r: 6, fill: '#1d4ed8' }}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
        <p className="text-xs text-gray-500 mt-2 text-center">
          Drift score represents semantic distance from the original version. Higher scores indicate more significant changes.
        </p>
      </div>

      {/* Significant Events */}
      {analysis.drift_events.length > 0 && (
        <div>
          <h4 className="font-medium text-gray-900 mb-4 flex items-center space-x-2">
            <AlertCircle className="w-4 h-4 text-yellow-500" />
            <span>Significant Drift Events ({analysis.drift_events.length})</span>
          </h4>
          
          <div className="space-y-3 max-h-64 overflow-y-auto">
            {analysis.drift_events.map((event, index) => (
              <div
                key={event.commit_hash}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedCommit?.commit_hash === event.commit_hash
                    ? 'border-primary-500 bg-primary-50'
                    : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => handleCommitClick(event)}
              >
                <div className="flex items-start justify-between">
                  <div className="flex-1">
                    <div className="flex items-center space-x-3 mb-2">
                      <GitCommit className="w-4 h-4 text-gray-400" />
                      <span className="font-mono text-sm text-gray-600">
                        {event.commit_hash.substring(0, 7)}
                      </span>
                      <span className="text-xs text-gray-500">
                        {new Date(event.timestamp).toLocaleDateString()}
                      </span>
                    </div>
                    
                    <p className="text-sm text-gray-900 mb-2">
                      {event.commit_message}
                    </p>
                    
                    <div className="flex items-center space-x-4 text-xs text-gray-600">
                      <span>+{event.added_lines} lines</span>
                      <span>-{event.removed_lines} lines</span>
                      <span className="font-medium">
                        Drift: {event.drift_score.toFixed(3)}
                      </span>
                    </div>
                  </div>
                  
                  <div className={`px-2 py-1 text-xs font-medium rounded ${
                    event.drift_score > 0.7 
                      ? 'bg-red-100 text-red-800' 
                      : event.drift_score > 0.5 
                      ? 'bg-yellow-100 text-yellow-800'
                      : 'bg-orange-100 text-orange-800'
                  }`}>
                    {event.drift_score > 0.7 ? 'Major' : event.drift_score > 0.5 ? 'Significant' : 'Moderate'}
                  </div>
                </div>
                
                {selectedCommit?.commit_hash === event.commit_hash && (
                  <div className="mt-3 pt-3 border-t border-gray-200">
                    <p className="text-sm text-gray-700">
                      {event.description}
                    </p>
                  </div>
                )}
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Timeline Events */}
      <div className="mt-6 pt-6 border-t">
        <h4 className="font-medium text-gray-900 mb-4">Recent Changes</h4>
        <div className="space-y-2 max-h-48 overflow-y-auto">
          {timelineData.slice(-10).reverse().map((commit) => (
            <div
              key={commit.commit_hash}
              className="flex items-center justify-between p-2 hover:bg-gray-50 rounded text-sm"
            >
              <div className="flex items-center space-x-3">
                <GitCommit className="w-3 h-3 text-gray-400" />
                <span className="font-mono text-gray-600">{commit.shortHash}</span>
                <span className="text-gray-900 truncate max-w-xs">
                  {commit.commit_message.substring(0, 50)}...
                </span>
              </div>
              <div className="flex items-center space-x-2">
                <span className="text-xs text-gray-500">{commit.date}</span>
                <span className={`px-2 py-1 text-xs rounded ${
                  commit.is_significant 
                    ? 'bg-yellow-100 text-yellow-800 font-medium' 
                    : 'bg-gray-100 text-gray-600'
                }`}>
                  {commit.driftScore.toFixed(3)}
                </span>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}

export default DriftTimeline
