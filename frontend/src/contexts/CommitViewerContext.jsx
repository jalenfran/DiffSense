import { createContext, useContext, useState } from 'react'
import CommitViewer from '../components/CommitViewer'

const CommitViewerContext = createContext()

export const CommitViewerProvider = ({ children }) => {
    const [isOpen, setIsOpen] = useState(false)
    const [viewerConfig, setViewerConfig] = useState({
        repoId: null,
        commitHash: null,
        title: null
    })

    const openCommitViewer = (config) => {
        setViewerConfig({
            repoId: config.repoId,
            commitHash: config.commitHash,
            title: config.title || null
        })
        setIsOpen(true)
    }

    const closeCommitViewer = () => {
        setIsOpen(false)
        // Clear config after a short delay to allow for smooth closing
        setTimeout(() => {
            setViewerConfig({
                repoId: null,
                commitHash: null,
                title: null
            })
        }, 300)
    }

    // Convenience function for opening commits
    const openCommit = (repoId, commitHash, title = null) => {
        openCommitViewer({
            repoId,
            commitHash,
            title: title || `Commit ${commitHash.substring(0, 8)}`
        })
    }

    return (
        <CommitViewerContext.Provider value={{
            isOpen,
            openCommitViewer,
            closeCommitViewer,
            openCommit
        }}>
            {children}
            <CommitViewer
                isOpen={isOpen}
                onClose={closeCommitViewer}
                repoId={viewerConfig.repoId}
                commitHash={viewerConfig.commitHash}
                title={viewerConfig.title}
            />
        </CommitViewerContext.Provider>
    )
}

export const useCommitViewer = () => {
    const context = useContext(CommitViewerContext)
    if (!context) {
        throw new Error('useCommitViewer must be used within a CommitViewerProvider')
    }
    return context
}

export default CommitViewerContext
