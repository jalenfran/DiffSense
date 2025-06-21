/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#eff6ff',
          500: '#3b82f6',
          600: '#2563eb',
          700: '#1d4ed8',
        },
        drift: {
          low: '#10b981',
          medium: '#f59e0b', 
          high: '#ef4444',
        }
      }
    },
  },
  plugins: [
    require('@tailwindcss/forms'),
  ],
}
