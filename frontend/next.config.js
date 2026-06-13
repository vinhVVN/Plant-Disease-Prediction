/** @type {import('next').NextConfig} */
const nextConfig = {
  // Bỏ qua lỗi type checking lúc build (nếu có do dependency mới rc)
  typescript: {
    ignoreBuildErrors: true,
  },
  webpack: (config, { isServer }) => {
    // Ngăn Webpack cố gắng polyfill 'fs' khi dùng onnxruntime-web trên trình duyệt
    if (!isServer) {
      config.resolve.fallback = {
        ...config.resolve.fallback,
        fs: false,
      };
    }
    return config;
  },
};

module.exports = nextConfig;
