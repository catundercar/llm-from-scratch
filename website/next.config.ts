import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  output: "export",
  basePath: "/llm-from-scratch",
  images: {
    unoptimized: true,
  },
  trailingSlash: true,
};

export default nextConfig;
