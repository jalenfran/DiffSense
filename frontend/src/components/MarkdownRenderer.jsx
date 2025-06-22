import React, { useEffect } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'

const MarkdownRenderer = ({ content, className = "" }) => {
  useEffect(() => {
    // Dynamically import highlight.js theme based on dark mode
    const isDark = document.documentElement.classList.contains('dark')
    
    // Remove any existing highlight.js stylesheets
    const existingStyles = document.querySelectorAll('link[href*="highlight.js"]')
    existingStyles.forEach(style => style.remove())
    
    // Add the appropriate theme
    const link = document.createElement('link')
    link.rel = 'stylesheet'
    link.href = isDark 
      ? 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github-dark.min.css'
      : 'https://cdnjs.cloudflare.com/ajax/libs/highlight.js/11.9.0/styles/github.min.css'
    document.head.appendChild(link)
    
    return () => {
      // Cleanup on unmount
      const styles = document.querySelectorAll('link[href*="highlight.js"]')
      styles.forEach(style => style.remove())
    }
  }, [])

  const customComponents = {
    // Custom code block styling
    code({ node, inline, className, children, ...props }) {
      const match = /language-(\w+)/.exec(className || '')
      return !inline ? (
        <code
          className={`${className} block bg-gray-100 dark:bg-gray-800 rounded p-3 text-sm overflow-x-auto font-mono`}
          {...props}
        >
          {children}
        </code>
      ) : (
        <code
          className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800 dark:text-gray-200"
          {...props}
        >
          {children}
        </code>
      )
    },
    // Custom pre block styling
    pre({ children }) {
      return (
        <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto my-4 border border-gray-200 dark:border-gray-700">
          {children}
        </pre>
      )
    },
    // Custom heading styling
    h1({ children }) {
      return <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-2">{children}</h1>
    },
    h2({ children }) {
      return <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-1">{children}</h2>
    },
    h3({ children }) {
      return <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-gray-100">{children}</h3>
    },
    h4({ children }) {
      return <h4 className="text-base font-bold mb-2 text-gray-900 dark:text-gray-100">{children}</h4>
    },
    // Custom paragraph styling
    p({ children }) {
      return <p className="mb-3 text-gray-900 dark:text-gray-100 leading-relaxed">{children}</p>
    },
    // Custom list styling
    ul({ children }) {
      return <ul className="list-disc pl-6 mb-3 text-gray-900 dark:text-gray-100 space-y-1">{children}</ul>
    },
    ol({ children }) {
      return <ol className="list-decimal pl-6 mb-3 text-gray-900 dark:text-gray-100 space-y-1">{children}</ol>
    },
    li({ children }) {
      return <li className="leading-relaxed">{children}</li>
    },
    // Custom blockquote styling
    blockquote({ children }) {
      return (
        <blockquote className="border-l-4 border-blue-500 pl-4 py-2 mb-4 bg-blue-50 dark:bg-blue-900/20 text-gray-700 dark:text-gray-300 italic rounded-r">
          {children}
        </blockquote>
      )
    },
    // Custom link styling
    a({ href, children }) {
      return (
        <a
          href={href}
          target="_blank"
          rel="noopener noreferrer"
          className="text-blue-600 dark:text-blue-400 hover:underline font-medium"
        >
          {children}
        </a>
      )
    },
    // Custom table styling
    table({ children }) {
      return (
        <div className="overflow-x-auto mb-4 rounded-lg border border-gray-200 dark:border-gray-700">
          <table className="min-w-full">
            {children}
          </table>
        </div>
      )
    },
    thead({ children }) {
      return <thead className="bg-gray-50 dark:bg-gray-800">{children}</thead>
    },
    tbody({ children }) {
      return <tbody className="bg-white dark:bg-gray-900">{children}</tbody>
    },
    th({ children }) {
      return (
        <th className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 text-left font-semibold text-gray-900 dark:text-gray-100 text-sm">
          {children}
        </th>
      )
    },
    td({ children }) {
      return (
        <td className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 text-gray-900 dark:text-gray-100 text-sm">
          {children}
        </td>
      )
    },
    // Custom horizontal rule styling
    hr() {
      return <hr className="my-6 border-gray-300 dark:border-gray-600" />
    },
    // Custom strong/bold styling
    strong({ children }) {
      return <strong className="font-bold text-gray-900 dark:text-gray-100">{children}</strong>
    },
    // Custom emphasis/italic styling
    em({ children }) {
      return <em className="italic text-gray-900 dark:text-gray-100">{children}</em>
    },
    // Custom checkbox styling for task lists
    input({ type, checked, ...props }) {
      if (type === 'checkbox') {
        return (
          <input
            type="checkbox"
            checked={checked}
            readOnly
            className="mr-2 h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
            {...props}
          />
        )
      }
      return <input type={type} {...props} />
    }
  }

  return (
    <div className={`prose prose-gray dark:prose-invert max-w-none ${className}`}>
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlight]}
        components={customComponents}
      >
        {content}
      </ReactMarkdown>
    </div>
  )
}

export default MarkdownRenderer
