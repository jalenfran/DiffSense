// Export all portable file viewer components
export { default as PortableFileViewer } from './PortableFileViewer'
export { default as FileViewerModal } from './FileViewerModal'
export { default as FileContentRenderer } from './FileContentRenderer'
export { default as FileViewerDemo } from './FileViewerDemo'

// Export example components
export {
    InlineFileViewer,
    CompactFilePreview,
    FullSizeFileViewer,
    FileLink,
    CommitLink,
    DiffComparison,
    CustomFileViewer
} from './FileViewerExamples'

// Export context and hooks
export { FileViewerProvider, useFileViewer } from '../contexts/FileViewerContext'

// Usage example:
// import { PortableFileViewer, useFileViewer, FileLink } from './components/FileViewer'
