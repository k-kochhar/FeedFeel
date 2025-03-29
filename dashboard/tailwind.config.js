/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx}",
    "./src/components/**/*.{js,ts,jsx,tsx}",
    "./src/app/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#F7F7F8",
        text: "#202123",
        accent: {
          DEFAULT: "#F0D070",
          light: "#FFFBEA",
          medium: "#F5E6B3",
          dark: "#D4AF37",
        },
        navy: {
          50: "#f0f4f8",
          100: "#d9e2ec",
          200: "#bcccdc",
          300: "#9fb3c8",
          400: "#829ab1",
          500: "#627d98",
          600: "#486581",
          700: "#334e68",
          800: "#243b53",
          900: "#102a43",
          950: "#0a1929",
        },
        coral: {
          50: '#fff1f0',
          100: '#ffe4e1',
          200: '#ffccc7',
          300: '#ffa69e',
          400: '#ff7e71',
          500: '#ff5649',
          600: '#e93c30',
          700: '#c42e23',
          800: '#a3271f',
          900: '#882720',
        },
      },
      fontFamily: {
        sans: ["Inter", "Helvetica Neue", "Arial", "sans-serif"],
      },
      fontSize: {
        base: ['18px', '1.8'],
      },
      boxShadow: {
        soft: "0px 2px 12px rgba(0, 0, 0, 0.05)",
        medium: "0px 4px 16px rgba(0, 0, 0, 0.08)",
      },
      borderRadius: {
        'xl': '1rem',
        '2xl': '1.5rem',
      },
      spacing: {
        '18': '4.5rem',
        '72': '18rem',
        '84': '21rem',
      },
      typography: {
        DEFAULT: {
          css: {
            maxWidth: '65ch',
            color: '#202123',
            lineHeight: 1.8,
          },
        },
      },
      backgroundImage: {
        "gradient-radial": "radial-gradient(var(--tw-gradient-stops))",
        "gradient-conic":
          "conic-gradient(from 180deg at 50% 50%, var(--tw-gradient-stops))",
      },
    },
  },
  plugins: [],
}; 