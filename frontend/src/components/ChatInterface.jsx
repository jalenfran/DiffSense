import { useState, useRef, useEffect } from 'react'
import { 
  Send, 
  MessageSquare, 
  Bot, 
  User, 
  FileText, 
  GitCommit, 
  AlertTriangle, 
  CheckCircle, 
  Clock,
  Code,
  Search,
  ChevronDown,
  ChevronUp,
  Copy,
  ExternalLink,
  Maximize2,
  X
} from 'lucide-react'

const ChatMessage = ({ message, isUser, timestamp }) => {
  const [isExpanded, setIsExpanded] = useState(false)

  const formatTimestamp = (timestamp) => {
    return new Date(timestamp).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  const copyToClipboard = (text) => {
    navigator.clipboard.writeText(text)
  }
  const renderMessageContent = (content) => {
    if (typeof content === 'string') {
      return (
        <div className="max-w-none text-gray-900 dark:text-gray-100">
          <p className="whitespace-pre-wrap">{content}</p>
        </div>
      )
    }

    // Handle structured API responses
    if (content.type === 'query_response') {
      return (
        <div className="space-y-4">
          <div className="max-w-none text-gray-900 dark:text-gray-100">
            <p className="whitespace-pre-wrap">{content.response}</p>
          </div>
          
          {content.confidence && (
            <div className="flex items-center gap-2 text-sm">
              <span className="text-gray-600 dark:text-gray-400">Confidence:</span>
              <div className="flex items-center gap-1">
                <div className="w-20 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                  <div 
                    className="bg-blue-600 h-2 rounded-full"
                    style={{ width: `${content.confidence * 100}%` }}
                  />
                </div>
                <span className="text-gray-900 dark:text-white font-medium">
                  {Math.round(content.confidence * 100)}%
                </span>
              </div>
            </div>
          )}

          {content.sources && content.sources.length > 0 && (
            <div className="bg-gray-50 dark:bg-gray-800 rounded-lg p-3">
              <button
                onClick={() => setIsExpanded(!isExpanded)}
                className="flex items-center gap-2 text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                <FileText className="w-4 h-4" />
                Sources ({content.sources.length})
                {isExpanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
              </button>
              
              {isExpanded && (
                <div className="space-y-2">
                  {content.sources.map((source, index) => (
                    <div key={index} className="flex items-center justify-between p-2 bg-white dark:bg-gray-700 rounded border">
                      <div className="flex items-center gap-2 min-w-0">
                        {source.type === 'file' ? <FileText className="w-4 h-4 text-blue-500" /> : <GitCommit className="w-4 h-4 text-green-500" />}
                        <span className="text-sm truncate font-mono">{source.path}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-gray-500 dark:text-gray-400">
                          {Math.round(source.relevance * 100)}% match
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )
    }

    if (content.type === 'commit_analysis') {
      return (
        <div className="space-y-4">
          <div className="flex items-center gap-2">
            <GitCommit className="w-5 h-5 text-blue-500" />
            <span className="font-medium">Commit Analysis</span>
            <span className="text-sm text-gray-500 dark:text-gray-400 font-mono">
              {content.commit_hash?.substring(0, 7)}
            </span>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-sm text-gray-600 dark:text-gray-400">Risk Score:</span>
            <div className="flex items-center gap-2">
              <div className="w-24 bg-gray-200 dark:bg-gray-700 rounded-full h-2">
                <div 
                  className={`h-2 rounded-full ${
                    content.overall_risk_score > 0.7 ? 'bg-red-500' : 
                    content.overall_risk_score > 0.4 ? 'bg-yellow-500' : 'bg-green-500'
                  }`}
                  style={{ width: `${content.overall_risk_score * 100}%` }}
                />
              </div>
              <span className="font-medium">
                {Math.round(content.overall_risk_score * 100)}%
              </span>
            </div>
          </div>

          {content.breaking_changes && content.breaking_changes.length > 0 && (
            <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <AlertTriangle className="w-4 h-4 text-red-500" />
                <span className="font-medium text-red-700 dark:text-red-300">
                  Breaking Changes ({content.breaking_changes.length})
                </span>
              </div>
              <div className="space-y-2">
                {content.breaking_changes.map((change, index) => (
                  <div key={index} className="bg-white dark:bg-gray-800 rounded p-2">
                    <div className="flex items-center gap-2 mb-1">
                      <span className={`px-2 py-1 text-xs rounded-full ${
                        change.risk_level === 'high' ? 'bg-red-100 text-red-700 dark:bg-red-900 dark:text-red-300' :
                        change.risk_level === 'medium' ? 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900 dark:text-yellow-300' :
                        'bg-green-100 text-green-700 dark:bg-green-900 dark:text-green-300'
                      }`}>
                        {change.risk_level}
                      </span>
                      <span className="text-sm font-medium">{change.change_type}</span>
                    </div>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mb-1">
                      {change.description}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-500 font-mono">
                      {change.file_path}
                    </p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {content.claude_analysis?.content && (
            <div className="bg-blue-50 dark:bg-blue-900/20 rounded-lg p-3">
              <div className="flex items-center gap-2 mb-2">
                <Bot className="w-4 h-4 text-blue-500" />
                <span className="font-medium text-blue-700 dark:text-blue-300">AI Analysis</span>
              </div>
              <p className="text-sm text-gray-700 dark:text-gray-300 whitespace-pre-wrap">
                {content.claude_analysis.content}
              </p>
              {content.claude_analysis.suggestions && (
                <div className="mt-2 space-y-1">
                  {content.claude_analysis.suggestions.map((suggestion, index) => (
                    <div key={index} className="flex items-start gap-2">
                      <CheckCircle className="w-3 h-3 text-green-500 mt-0.5 flex-shrink-0" />
                      <span className="text-xs text-gray-600 dark:text-gray-400">{suggestion}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>
          )}
        </div>
      )
    }

    return <div className="text-gray-500 dark:text-gray-400">Unsupported message type</div>
  }

  return (
    <div className={`flex gap-3 ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`flex gap-3 max-w-[80%] ${isUser ? 'flex-row-reverse' : 'flex-row'}`}>
        <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
          isUser ? 'bg-blue-500' : 'bg-gray-600 dark:bg-gray-500'
        }`}>
          {isUser ? (
            <User className="w-4 h-4 text-white" />
          ) : (
            <Bot className="w-4 h-4 text-white" />
          )}
        </div>
        
        <div className={`rounded-lg px-4 py-3 ${
          isUser 
            ? 'bg-blue-500 text-white' 
            : 'bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700'
        }`}>
          {renderMessageContent(message)}
          <div className={`flex items-center justify-between mt-2 pt-2 border-t ${
            isUser 
              ? 'border-blue-400' 
              : 'border-gray-200 dark:border-gray-700'
          }`}>
            <span className={`text-xs ${
              isUser 
                ? 'text-blue-100' 
                : 'text-gray-500 dark:text-gray-400'
            }`}>
              {formatTimestamp(timestamp)}
            </span>
            <button
              onClick={() => copyToClipboard(typeof message === 'string' ? message : JSON.stringify(message))}
              className={`text-xs p-1 rounded hover:bg-opacity-20 ${
                isUser 
                  ? 'text-blue-100 hover:bg-white' 
                  : 'text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-600'
              }`}
              title="Copy message"
            >
              <Copy className="w-3 h-3" />
            </button>
          </div>
        </div>
      </div>
    </div>
  )
}

function ChatInterface({ 
  repoId, 
  onSendMessage, 
  messages, 
  isLoading, 
  onFileSelect, 
  selectedFiles, 
  availableFiles, 
  isMaximized = false, 
  onToggleMaximize 
}) {
  const [inputMessage, setInputMessage] = useState('')
  const [showFileSelector, setShowFileSelector] = useState(false)
  const messagesEndRef = useRef(null)
  const inputRef = useRef(null)

  // Ensure availableFiles is always an array
  const safeAvailableFiles = Array.isArray(availableFiles) ? availableFiles : []
  const safeSelectedFiles = Array.isArray(selectedFiles) ? selectedFiles : []
  const safeMessages = Array.isArray(messages) ? messages : []
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [safeMessages])

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isLoading) return

    const messageText = inputMessage.trim()
    setInputMessage('')
    
    await onSendMessage(messageText, safeSelectedFiles)
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSendMessage()
    }
  }

  const quickPrompts = [
    {
      label: "Analyze recent changes",
      prompt: "What are the most significant changes in the recent commits?",
      icon: <GitCommit className="w-4 h-4" />
    },
    {
      label: "Find breaking changes",
      prompt: "Are there any breaking changes that could affect users?",
      icon: <AlertTriangle className="w-4 h-4" />
    },
    {
      label: "Code structure overview",
      prompt: "Can you give me an overview of the main components and structure of this project?",
      icon: <Code className="w-4 h-4" />
    },
    {
      label: "Security concerns",
      prompt: "Are there any security vulnerabilities or concerns in recent changes?",
      icon: <Search className="w-4 h-4" />    }
  ]

  return (
    <div className={`flex flex-col h-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg transition-all duration-300 ease-in-out ${
      isMaximized ? 'fixed inset-2 z-50 shadow-2xl' : ''
    }`}>
      {/* Chat Header */}
      <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex items-center gap-2">
          <MessageSquare className="w-5 h-5 text-blue-500" />
          <h3 className="font-medium text-gray-900 dark:text-white">Repository Chat</h3>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFileSelector(!showFileSelector)}
            className={`text-sm px-3 py-1 rounded-lg transition-colors ${
              safeSelectedFiles.length > 0
                ? 'bg-blue-100 text-blue-700 dark:bg-blue-900 dark:text-blue-300'
                : 'bg-gray-100 text-gray-600 dark:bg-gray-700 dark:text-gray-300'
            }`}
          >
            {safeSelectedFiles.length > 0 ? `${safeSelectedFiles.length} files selected` : 'Select files'}
          </button>
          {onToggleMaximize && (
            <button
              onClick={onToggleMaximize}
              className="p-2 text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-white hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
              title={isMaximized ? 'Minimize chat' : 'Maximize chat'}
            >
              {isMaximized ? (
                <X className="w-5 h-5" />
              ) : (
                <Maximize2 className="w-5 h-5" />
              )}
            </button>
          )}
        </div>
      </div>

      {/* File Selector */}
      {showFileSelector && (
        <div className="p-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <div className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Select files to focus the conversation:
          </div>          <div className="max-h-32 overflow-y-auto space-y-1">
            {safeAvailableFiles.length > 0 ? (
              safeAvailableFiles.map((file, index) => (
                <label key={index} className="flex items-center gap-2 text-sm">
                  <input
                    type="checkbox"
                    checked={safeSelectedFiles.includes(file)}
                    onChange={(e) => {
                      if (e.target.checked) {
                        onFileSelect([...safeSelectedFiles, file])
                      } else {
                        onFileSelect(safeSelectedFiles.filter(f => f !== file))
                      }
                    }}
                    className="rounded border-gray-300 dark:border-gray-600"
                  />
                  <span className="font-mono text-gray-600 dark:text-gray-400 truncate">
                    {file}
                  </span>
                </label>
              ))
            ) : (
              <div className="text-sm text-gray-500 dark:text-gray-400 text-center py-2">
                No files available. Repository may still be loading.
              </div>
            )}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4">
        {safeMessages.length === 0 ? (
          <div className="text-center py-8">
            <Bot className="w-12 h-12 text-gray-400 dark:text-gray-500 mx-auto mb-3" />
            <p className="text-gray-600 dark:text-gray-400 mb-4">
              Start a conversation about your repository
            </p>
            
            {/* Quick Prompts */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 max-w-lg mx-auto">
              {quickPrompts.map((prompt, index) => (
                <button
                  key={index}
                  onClick={() => setInputMessage(prompt.prompt)}
                  className="flex items-center gap-2 p-3 text-left bg-gray-50 dark:bg-gray-700 hover:bg-gray-100 dark:hover:bg-gray-600 rounded-lg transition-colors"
                >
                  {prompt.icon}
                  <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                    {prompt.label}
                  </span>
                </button>
              ))}
            </div>
          </div>        ) : (
          <>
            {safeMessages.map((msg, index) => (
              <ChatMessage
                key={index}
                message={msg.content}
                isUser={msg.isUser}
                timestamp={msg.timestamp}
              />
            ))}
            
            {isLoading && (
              <div className="flex gap-3 justify-start">
                <div className="w-8 h-8 rounded-full bg-gray-600 dark:bg-gray-500 flex items-center justify-center flex-shrink-0">
                  <Bot className="w-4 h-4 text-white" />
                </div>
                <div className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg px-4 py-3">
                  <div className="flex items-center gap-2">
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-blue-500"></div>
                    <span className="text-sm text-gray-600 dark:text-gray-400">Thinking...</span>
                  </div>
                </div>
              </div>
            )}
          </>
        )}
        <div ref={messagesEndRef} />
      </div>      {/* Input */}
      <div className="p-4 border-t border-gray-200 dark:border-gray-700">
        <div className="flex gap-2 items-start">
          <div className="flex-1">
            <textarea
              ref={inputRef}
              value={inputMessage}
              onChange={(e) => setInputMessage(e.target.value)}
              onKeyPress={handleKeyPress}
              placeholder="Ask about this repository..."
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent resize-none bg-white dark:bg-gray-700 text-gray-900 dark:text-white placeholder-gray-500 dark:placeholder-gray-400"
              rows={1}
              style={{ minHeight: '42px', maxHeight: '120px' }}
              onInput={(e) => {
                e.target.style.height = 'auto'
                e.target.style.height = e.target.scrollHeight + 'px'
              }}
            />
          </div>
          <button
            onClick={handleSendMessage}
            disabled={!inputMessage.trim() || isLoading}
            className="h-[42px] px-4 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-300 dark:disabled:bg-gray-600 text-white rounded-lg transition-colors disabled:cursor-not-allowed flex items-center gap-2 flex-shrink-0"
          >
            <Send className="w-4 h-4" />
          </button>
        </div>
        {safeSelectedFiles.length > 0 && (
          <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
            Context: {safeSelectedFiles.length} file{safeSelectedFiles.length !== 1 ? 's' : ''} selected
          </div>
        )}
      </div>
    </div>
  )
}

// Default props for the ChatInterface component
ChatInterface.defaultProps = {
  messages: [],
  selectedFiles: [],
  availableFiles: [],
  isLoading: false,
  isMaximized: false,
  onToggleMaximize: null
}

export default ChatInterface
