import React, { useEffect, useState, useCallback } from 'react'
import ReactMarkdown from 'react-markdown'
import remarkGfm from 'remark-gfm'
import rehypeHighlight from 'rehype-highlight'
import { useFileViewer } from '../contexts/FileViewerContext'
import { diffSenseAPI } from '../services/api'

const MarkdownRenderer = ({ content, className = "", repoId = null }) => {
    const { openFileViewer } = useFileViewer();
    const [repositoryFiles, setRepositoryFiles] = useState([]);
    const [filesLoaded, setFilesLoaded] = useState(false);    // Load repository files when repoId changes
    useEffect(() => {
        if (repoId && !filesLoaded) {
            const loadRepositoryFiles = async () => {
                try {
                    const filesData = await diffSenseAPI.getRepositoryFiles(repoId);
                    setRepositoryFiles(filesData.files || []);
                    setFilesLoaded(true);                    // DEBUG: File loading
                    console.log('=== REPO FILES LOADED ===');
                    console.log('RepoId:', repoId);
                    console.log('Files count:', filesData.files?.length || 0);
                    console.log('Sample files:', filesData.files?.slice(0, 5) || []);
                    console.log('First file type:', typeof filesData.files?.[0]);
                    console.log('First file structure:', filesData.files?.[0]);
                } catch (error) {
                    console.error('Failed to load repository files:', error);
                    setRepositoryFiles([]);
                    setFilesLoaded(true);
                }
            };
            loadRepositoryFiles();
        }
    }, [repoId, filesLoaded]);

    // Reset files when repoId changes
    useEffect(() => {
        setFilesLoaded(false);
        setRepositoryFiles([]);
    }, [repoId]);

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
    }, [])    // Find the actual file path in the repository that matches the mentioned filename
    const findFileInRepository = useCallback((filename) => {
        // DEBUG: File search start
        console.log('=== FILE SEARCH DEBUG ===');
        console.log('Looking for filename:', filename);
        console.log('Repository files available:', repositoryFiles.length);
        console.log('Files loaded:', filesLoaded);

        if (!filename || !repositoryFiles.length) {
            console.log('Early return: no filename or no repo files');
            return null;
        }

        // Extract just the filename from the input (in case it has a partial path)
        const targetFilename = filename.split('/').pop();
        console.log('Target filename extracted:', targetFilename);        // Look for files that end with this filename
        const matches = repositoryFiles.filter(file => {
            // Handle both string and object formats
            const filePath = typeof file === 'string' ? file : (file.path || file.name || '');
            if (!filePath) return false;
            
            const repoFilename = filePath.split('/').pop();
            return repoFilename === targetFilename;
        });

        console.log('Exact matches found:', matches);        if (matches.length === 1) {
            console.log('Using exact match:', matches[0]);
            return typeof matches[0] === 'string' ? matches[0] : (matches[0].path || matches[0].name || null);
        } else if (matches.length > 1) {
            // Multiple matches - prefer shorter paths or common directories
            const sorted = matches.sort((a, b) => {
                // Get file paths
                const aPath = typeof a === 'string' ? a : (a.path || a.name || '');
                const bPath = typeof b === 'string' ? b : (b.path || b.name || '');
                
                // Prefer files in src/, common directories
                const aHasCommonDir = aPath.includes('src/') || aPath.includes('lib/') || aPath.includes('app/');
                const bHasCommonDir = bPath.includes('src/') || bPath.includes('lib/') || bPath.includes('app/');
                
                if (aHasCommonDir && !bHasCommonDir) return -1;
                if (!aHasCommonDir && bHasCommonDir) return 1;
                
                // Otherwise prefer shorter paths
                return aPath.length - bPath.length;
            });
            console.log('Multiple matches, sorted:', sorted);
            console.log('Using first sorted match:', sorted[0]);
            return typeof sorted[0] === 'string' ? sorted[0] : (sorted[0].path || sorted[0].name || null);
        }        // No exact match - try partial matching for common patterns
        const partialMatches = repositoryFiles.filter(file => {
            const filePath = typeof file === 'string' ? file : (file.path || file.name || '');
            return filePath.toLowerCase().includes(targetFilename.toLowerCase());
        });

        console.log('Partial matches found:', partialMatches);

        if (partialMatches.length > 0) {
            console.log('Using partial match:', partialMatches[0]);
            return typeof partialMatches[0] === 'string' ? partialMatches[0] : (partialMatches[0].path || partialMatches[0].name || null);
        }

        console.log('No matches found for:', filename);
        console.log('=== END FILE SEARCH DEBUG ===');
        return null;
    }, [repositoryFiles]); const handleFileClick = (filePath) => {
        console.log('=== FILE CLICK DEBUG ===');
        console.log('Clicked file path:', filePath);
        console.log('RepoId:', repoId);
        console.log('openFileViewer available:', !!openFileViewer);

        if (!repoId || !openFileViewer) return;

        // Try to find the actual file path in the repository
        const actualFilePath = findFileInRepository(filePath);
        console.log('Resolved actual file path:', actualFilePath);

        if (actualFilePath) {
            console.log('Opening file viewer with resolved path:', actualFilePath);
            openFileViewer({
                repoId,
                filePath: actualFilePath,
                commitHash: null,
                viewMode: 'file'
            });
        } else {
            console.log('File not found in repository, using fallback path:', filePath);
            openFileViewer({
                repoId,
                filePath,
                commitHash: null,
                viewMode: 'file'
            });
        }
        console.log('=== END FILE CLICK DEBUG ===');
    };

    const handleCommitClick = (commitHash) => {
        if (repoId && openFileViewer) {
            openFileViewer({
                repoId,
                filePath: null,
                commitHash,
                viewMode: 'commit'
            })
        }
    }; const linkifyText = (text) => {
        if (!repoId || typeof text !== 'string') return text;

        // File path regex - matches paths like src/file.js, .gitignore, etc.
        const filePathRegex = /((?:[a-zA-Z0-9_.-]+\/)*\.?[a-zA-Z0-9_.-]+\.[a-zA-Z0-9]+)/g;
        // Commit hash regex - matches 7-40 character hex strings
        const commitHashRegex = /\b([a-f0-9]{7,40})\b/g;

        const parts = [];
        let lastIndex = 0;
        const matches = [];

        // Find file paths
        let match;
        while ((match = filePathRegex.exec(text)) !== null) {
            if (!match[0].includes('://') && !match[0].includes('@')) {
                matches.push({
                    index: match.index,
                    length: match[0].length,
                    text: match[0],
                    type: 'file'
                });
            }
        }

        // Find commit hashes
        filePathRegex.lastIndex = 0;
        while ((match = commitHashRegex.exec(text)) !== null) {
            // Skip if this overlaps with a file path
            const overlaps = matches.some(m =>
                match.index < m.index + m.length && match.index + match[0].length > m.index
            );
            if (!overlaps) {
                matches.push({
                    index: match.index,
                    length: match[0].length,
                    text: match[0],
                    type: 'commit'
                });
            }
        }        // Sort matches by position
        matches.sort((a, b) => a.index - b.index);

        // Build result with linkified parts
        matches.forEach((match, i) => {
            // Add text before this match
            if (match.index > lastIndex) {
                parts.push(text.substring(lastIndex, match.index));
            }

            // Add the linkified button
            if (match.type === 'file') {
                parts.push(
                    <button
                        key={`file-${match.index}-${i}`}
                        onClick={() => handleFileClick(match.text)}
                        className="text-blue-600 dark:text-blue-400 hover:text-blue-800 dark:hover:text-blue-300 underline underline-offset-2 decoration-1 hover:decoration-2 transition-all duration-150 font-mono text-sm bg-blue-50 dark:bg-blue-900/20 px-1 py-0.5 rounded mx-0.5"
                        title={`View file: ${match.text}`}
                    >
                        {match.text}
                    </button>
                );
            } else {
                parts.push(
                    <button
                        key={`commit-${match.index}-${i}`}
                        onClick={() => handleCommitClick(match.text)}
                        className="text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300 underline underline-offset-2 decoration-1 hover:decoration-2 transition-all duration-150 font-mono text-sm bg-green-50 dark:bg-green-900/20 px-1 py-0.5 rounded mx-0.5"
                        title={`View commit: ${match.text}`}
                    >
                        {match.text}
                    </button>
                );
            }

            lastIndex = match.index + match.length;
        });

        // Add remaining text
        if (lastIndex < text.length) {
            parts.push(text.substring(lastIndex));
        }

        // Return the parts if we have any matches, otherwise return original text
        return matches.length > 0 ? parts : text;
    }; const customComponents = {
        // Custom paragraph styling - this is where most text content goes
        p({ children }) {

            // Extract all text content from children to process as a whole
            const extractText = (child) => {
                if (typeof child === 'string') {
                    return child;
                } else if (React.isValidElement(child) && child.props && child.props.children) {
                    // Handle inline elements like <code>
                    if (typeof child.props.children === 'string') {
                        return child.props.children;
                    } else if (Array.isArray(child.props.children)) {
                        return child.props.children.map(extractText).join('');
                    }
                }
                return '';
            }; const fullText = React.Children.toArray(children).map(extractText).join('');

            // If we have linkifiable content, process the entire paragraph
            const linkifiedResult = linkifyText(fullText);
            if (Array.isArray(linkifiedResult) && linkifiedResult.length > 1) {
                return <p className="mb-3 text-gray-900 dark:text-gray-100 leading-relaxed">{linkifiedResult}</p>;
            }

            // Otherwise, process children individually (fallback)
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };

            const processedChildren = React.Children.map(children, processChild);
            return <p className="mb-3 text-gray-900 dark:text-gray-100 leading-relaxed">{processedChildren}</p>
        },
        // Custom list item styling - also processes text
        li({ children }) {
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };

            const processedChildren = React.Children.map(children, processChild);
            return <li className="leading-relaxed">{processedChildren}</li>
        },// Custom code block styling
        code({ node, inline, className, children, ...props }) {
            const match = /language-(\w+)/.exec(className || '')

            if (!inline) {
                // Block code - don't linkify
                return (
                    <code
                        className={`${className} block bg-gray-100 dark:bg-gray-800 rounded p-3 text-sm overflow-x-auto font-mono`}
                        {...props}
                    >
                        {children}
                    </code>
                )
            } else {
                // Inline code - process for linkification
                const processChild = (child) => {
                    if (typeof child === 'string') {
                        return linkifyText(child);
                    }
                    return child;
                };

                const processedChildren = React.Children.map(children, processChild);

                return (
                    <code
                        className="bg-gray-100 dark:bg-gray-800 px-1.5 py-0.5 rounded text-sm font-mono text-gray-800 dark:text-gray-200"
                        {...props}
                    >
                        {processedChildren}
                    </code>
                )
            }
        },
        // Custom pre block styling
        pre({ children }) {
            return (
                <pre className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 overflow-x-auto my-4 border border-gray-200 dark:border-gray-700">
                    {children}
                </pre>
            )
        },        // Custom heading styling
        h1({ children }) {
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };
            const processedChildren = React.Children.map(children, processChild);
            return <h1 className="text-2xl font-bold mb-4 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-2">{processedChildren}</h1>
        },
        h2({ children }) {
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };
            const processedChildren = React.Children.map(children, processChild);
            return <h2 className="text-xl font-bold mb-3 text-gray-900 dark:text-gray-100 border-b border-gray-200 dark:border-gray-700 pb-1">{processedChildren}</h2>
        },
        h3({ children }) {
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };
            const processedChildren = React.Children.map(children, processChild);
            return <h3 className="text-lg font-bold mb-2 text-gray-900 dark:text-gray-100">{processedChildren}</h3>
        },
        h4({ children }) {
            const processChild = (child) => {
                if (typeof child === 'string') {
                    return linkifyText(child);
                }
                return child;
            };
            const processedChildren = React.Children.map(children, processChild);
            return <h4 className="text-base font-bold mb-2 text-gray-900 dark:text-gray-100">{processedChildren}</h4>
        },
        // Custom list styling
        ul({ children }) {
            return <ul className="list-disc pl-6 mb-3 text-gray-900 dark:text-gray-100 space-y-1">{children}</ul>
        }, ol({ children }) {
            return <ol className="list-decimal pl-6 mb-3 text-gray-900 dark:text-gray-100 space-y-1">{children}</ol>
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
