import { resolve } from 'node:path';
import { defineConfig } from 'vite';

// Two pages: the photo walk, and the 3D walk (three.js stays out of the photo walk).
export default defineConfig({
  build: {
    rollupOptions: {
      input: { walk: resolve(import.meta.dirname, 'index.html'), three: resolve(import.meta.dirname, '3d.html') },
    },
  },
});
