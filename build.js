const esbuild = require('esbuild');

esbuild.build({
  entryPoints: ['src/profile.ts'],
  bundle: true,
  outfile: 'dist/index.js',
  platform: 'node',
  target: 'node14',
  external: ['@aws-sdk/*'],
  format: 'cjs',
}).catch(() => process.exit(1));