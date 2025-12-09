/* eslint-env node */
/** @type {import('tailwindcss').Config} */
module.exports = {
  content: ["./index.html", "./src/**/*.{js,jsx,ts,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50: "#f2fbff",
          100: "#e6f7ff",
          300: "#7fd3ff",
          500: "#38bdf8", // primary-ish
          700: "#0ea5a4",
        },
      },
      maxWidth: {
        "screen-sm": "680px",
      },
    },
  },
  plugins: [],
};
