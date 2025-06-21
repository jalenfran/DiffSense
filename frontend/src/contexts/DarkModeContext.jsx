import { createContext, useContext, useState, useEffect } from 'react'

const DarkModeContext = createContext()

export const useDarkMode = () => {
    const context = useContext(DarkModeContext)
    if (!context) {
        throw new Error('useDarkMode must be used within a DarkModeProvider')
    }
    return context
}

export const DarkModeProvider = ({ children }) => {
    const [darkMode, setDarkMode] = useState(() => {
        const saved = localStorage.getItem('diffsense-dark-mode')
        return saved ? JSON.parse(saved) : false
    })

    useEffect(() => {
        localStorage.setItem('diffsense-dark-mode', JSON.stringify(darkMode))

        // Temporarily disable transitions to prevent phased animations
        document.documentElement.style.setProperty('--disable-transitions', '1')

        // Add or remove dark class from html element
        if (darkMode) {
            document.documentElement.classList.add('dark')
        } else {
            document.documentElement.classList.remove('dark')
        }

        // Re-enable transitions after a brief delay
        setTimeout(() => {
            document.documentElement.style.removeProperty('--disable-transitions')
        }, 50)
    }, [darkMode])

    const toggleDarkMode = () => {
        setDarkMode(prev => !prev)
    }

    return (
        <DarkModeContext.Provider value={{ darkMode, toggleDarkMode }}>
            {children}
        </DarkModeContext.Provider>
    )
}
