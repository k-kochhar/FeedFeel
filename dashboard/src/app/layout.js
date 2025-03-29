import { Inter } from 'next/font/google'
import './globals.css'

const inter = Inter({ subsets: ['latin'], display: 'swap' })

export const metadata = {
  title: 'FeedFeel',
  description: 'Control your camera feed with elegance',
}

export default function RootLayout({ children }) {
  return (
    <html lang="en" className={inter.className}>
      <body className="bg-background min-h-screen">{children}</body>
    </html>
  )
}
