function LoadingSpinner() {
    return (
        <div className="min-h-screen flex items-center justify-center bg-gray-50">
            <div className="flex flex-col items-center gap-4">
                <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
                <p className="text-gray-600 font-medium">Loading DiffSense...</p>
            </div>
        </div>
    )
}

export default LoadingSpinner
