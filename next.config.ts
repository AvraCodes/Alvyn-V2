import type { NextConfig } from "next"

const nextConfig: NextConfig = {
  // Enable experimental features if needed
  experimental: {
    // serverActions: true,
  },
  // Silence incorrect workspace root warning by explicitly setting Turbopack root
  turbopack: {
    root: "/Users/avra/Proj2",
  },
  // Configure rewrites for API proxy (alternative to route handler)
  async rewrites() {
    return [
      // Direct proxy for /api/backend/* routes to FastAPI
      {
        source: "/api/backend/:path*",
        destination: `${process.env.BACKEND_URL || "http://localhost:8000"}/:path*`,
      },
    ]
  },
}

export default nextConfig
