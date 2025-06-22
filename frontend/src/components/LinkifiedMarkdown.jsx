import MarkdownRenderer from './MarkdownRenderer'

const LinkifiedMarkdown = ({ content, repoId, className = "" }) => {
    // Pass repoId directly to MarkdownRenderer so it can handle linkification internally
    return <MarkdownRenderer content={content} className={className} repoId={repoId} />
}

export default LinkifiedMarkdown
