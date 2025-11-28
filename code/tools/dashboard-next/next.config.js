/** @type {import('next').NextConfig} */
const backendHost = process.env.BACKEND_HOST || process.env.NEXT_PUBLIC_BACKEND_HOST || '127.0.0.1';
const backendPort = process.env.BACKEND_PORT || process.env.NEXT_PUBLIC_BACKEND_PORT || 6970;

const nextConfig = {
  // Proxy API requests to Python backend
  async rewrites() {
    return [
      {
        source: '/api/:path*',
        destination: `http://${backendHost}:${backendPort}/api/:path*`,
      },
    ];
  },
};

module.exports = nextConfig;
