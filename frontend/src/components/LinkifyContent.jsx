import React from 'react';
import { useFileViewer } from '../contexts/FileViewerContext';
import { useCommitViewer } from '../contexts/CommitViewerContext';

const LinkifyContent = ({ children, repoId }) => {
    const { openFileWithCommitNavigation } = useFileViewer();
    const { openCommit } = useCommitViewer();

    const handleFileClick = (filePath) => {
        if (repoId) {
            console.log('LinkifyContent: Opening file viewer for file with commit navigation', { repoId, filePath })
            openFileWithCommitNavigation(repoId, filePath)
        }
    };

    const handleCommitClick = (commitHash) => {
        if (repoId) {
            console.log('LinkifyContent: Opening commit viewer for commit', { repoId, commitHash })
            openCommit(repoId, commitHash)
        }
    };const processTextNode = (text) => {
        if (typeof text !== 'string') return text;

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
        }        // Find commit hashes
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
        }

        // Sort matches by position
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
                        key={`file-${i}`}
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
                        key={`commit-${i}`}
                        onClick={() => handleCommitClick(match.text)}
                        className="text-green-600 dark:text-green-400 hover:text-green-800 dark:hover:text-green-300 underline underline-offset-2 decoration-1 hover:decoration-2 transition-all duration-150 font-mono text-sm bg-green-50 dark:bg-green-900/20 px-1 py-0.5 rounded mx-0.5"
                        title={`View commit: ${match.text}`}
                    >
                        {match.text}
                    </button>
                );
            }

            lastIndex = match.index + match.length;
        });        // Add remaining text
        if (lastIndex < text.length) {
            parts.push(text.substring(lastIndex));
        }

        // Return the parts if we have any matches, otherwise return original text
        return matches.length > 0 ? parts : text;
    };
    const processChildren = (children) => {

        if (typeof children === 'string') {
            const result = processTextNode(children);
            return result;
        } if (React.isValidElement(children)) {

            // Don't process code blocks and similar
            if (children.type === 'code' || children.type === 'pre') {
                return children;
            }

            // Process children recursively
            const processedChildren = React.Children.map(children.props.children, processChildren);

            return React.cloneElement(children, {
                ...children.props,
                children: processedChildren
            });
        }

        if (Array.isArray(children)) {
            return children.map((child, index) =>
                React.isValidElement(child) ?
                    React.cloneElement(processChildren(child), { key: child.key || index }) :
                    processChildren(child)
            );
        }

        return children;
    };

    if (!repoId) {
        return children;
    } const processed = processChildren(children);
    return <>{processed}</>;
};

export default LinkifyContent;
