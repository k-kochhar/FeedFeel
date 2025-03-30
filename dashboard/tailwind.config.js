/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
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
          50: "#f0f5fa",
          100: "#d0e1f5",
          200: "#b0cef0",
          300: "#90bbe5",
          400: "#70a8db",
          500: "#5095d0",
          600: "#4082c0",
          700: "#316eb5",
          800: "#215ca0",
          900: "#114990",
          950: "#003780",
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
      animation: {
        'spin-slow': 'spin 3s linear infinite',
      },
    },
  },
  plugins: [],
}; 