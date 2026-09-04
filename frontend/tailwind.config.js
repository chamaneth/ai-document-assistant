/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        cyber: {
          bg: "#0F0F11",
          card: "#16161A",
          sidebar: "#121215",
          accent: "#3B82F6",
          accentGlow: "#60A5FA",
          cyan: "#00F0FF",
          border: "rgba(255, 255, 255, 0.08)",
          borderGlow: "rgba(59, 130, 246, 0.4)",
          text: "#E2E8F0",
          muted: "#94A3B8",
        }
      },
      fontFamily: {
        sans: ['Inter', 'Outfit', 'sans-serif'],
        mono: ['Fira Code', 'JetBrains Mono', 'monospace'],
      },
      boxShadow: {
        'cyber-glow': '0 0 20px rgba(59, 130, 246, 0.25)',
        'cyan-glow': '0 0 20px rgba(0, 240, 255, 0.25)',
        'glass': '0 8px 32px 0 rgba(0, 0, 0, 0.37)',
      }
    },
  },
  plugins: [],
}
