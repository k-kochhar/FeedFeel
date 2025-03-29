import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], display: 'swap' })

export const metadata = {
  title: 'SixthSense',
  description: 'Transforming Vision into Touch',
  icons: {
    icon: '/favicon.svg',
  },
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body className="bg-background min-h-screen">{children}</body>
    </html>
  )
}
