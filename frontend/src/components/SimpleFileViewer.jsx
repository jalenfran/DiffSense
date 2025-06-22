import React from 'react'
import { X } from 'lucide-react'
import FileContentRenderer from './FileContentRenderer'

const FileViewer = ({
    isOpen,
    onClose,
    repoId,
    filePath,
    commitHash = null,
    title = null
}) => {
    if (!isOpen) return null

    const displayTitle = title || (filePath ? filePath.split('/').pop() : 'File Viewer')

    return (
        <div className="fixed inset-0 bg-black bg-opacity-50 dark:bg-black dark:bg-opacity-70 flex items-center justify-center z-50">
            <div className="bg-white dark:bg-gray-800 rounded-lg w-full h-full max-w-[95vw] max-h-[95vh] flex flex-col shadow-2xl">
                {/* Header */}
                <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
                    <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
                        {displayTitle}
                    </h2>
                    <button
                        onClick={onClose}
                        className="p-2 text-gray-600 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                    >
                        <X className="w-5 h-5" />
                    </button>
                </div>

                {/* Content */}
                <div className="flex-1 overflow-hidden">
                    <FileContentRenderer
                        repoId={repoId}
                        filePath={filePath}
                        commitHash={commitHash}
                        height="100%"
                        className="h-full border-0 rounded-none"
                        showHeader={false} // Header is handled by the modal
                    />
                </div>
            </div>
        </div>
    )
}

export default FileViewer
