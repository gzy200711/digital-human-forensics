import type { NextConfig } from "next";

const apiBaseUrl = (process.env.API_BASE_URL ?? "http://localhost:8000").replace(
  /\/+$/,
  "",
);

const nextConfig: NextConfig = {
  output: "standalone",
  experimental: {
    proxyClientMaxBodySize: "600mb",
  },
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: `${apiBaseUrl}/:path*`,
      },
      {
        source: "/v1/:path*",
        destination: `${apiBaseUrl}/v1/:path*`,
      },
    ];
  },
};

export default nextConfig;
