export default {
  webpack: (config, { isServer }) => {
    if (isServer) {
      config.externals.push({
        '@tensorflow/tfjs-node': 'commonjs @tensorflow/tfjs-node',
      });
    }
    return config;
  },
};
