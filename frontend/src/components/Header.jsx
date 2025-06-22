import { LogOut, User, GitBranch } from 'lucide-react'
import { diffSenseAPI } from '../services/api'

function Header({ user, onLogout }) {
    const handleLogout = async () => {
        try {
            await diffSenseAPI.logout()
        } catch (error) {
            console.error('Logout error:', error)
        } finally {
            onLogout()
        }
    }

    return (
        <header className="bg-white border-b border-gray-200 sticky top-0 z-40">
            <div className="px-6 py-4">
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="bg-primary-600 p-2 rounded-lg">
                            <GitBranch className="w-6 h-6 text-white" />
                        </div>
                        <h1 className="text-xl font-bold text-gray-900">DiffSense</h1>
                    </div>                    <div className="flex items-center gap-4">
                        <div className="flex items-center gap-3">
                            <div className="w-8 h-8 bg-gray-300 rounded-full flex items-center justify-center">
                                {user.github_avatar ? (
                                    <img
                                        src={user.github_avatar}
                                        alt={user.display_name || user.github_username}
                                        className="w-8 h-8 rounded-full"
                                    />
                                ) : (
                                    <User className="w-4 h-4 text-gray-600" />
                                )}
                            </div>
                            <div className="text-sm">
                                <p className="font-medium text-gray-900">
                                    {user.display_name || user.github_username}
                                </p>
                                <p className="text-gray-500">@{user.github_username}</p>
                            </div>
                        </div>

                        <button
                            onClick={handleLogout}
                            className="flex items-center gap-2 text-gray-600 hover:text-gray-900 px-3 py-2 rounded-lg hover:bg-gray-100 transition-colors"
                        >
                            <LogOut className="w-4 h-4" />
                            <span className="text-sm font-medium">Logout</span>
                        </button>
                    </div>
                </div>
            </div>
        </header>
    )
}

export default Header
