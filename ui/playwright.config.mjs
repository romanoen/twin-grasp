import {existsSync} from 'node:fs';
import {defineConfig} from '@playwright/test';

// Uses the local Chrome if present, otherwise Playwright's bundled Chromium (npx playwright install chromium).
const chrome = '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome';
const port = Number(process.env.PORT || 4173);

export default defineConfig({
  testDir: './browser-tests',
  workers: 1,
  reporter: 'list',
  webServer: {command: 'node server.mjs', url: `http://127.0.0.1:${port}`, reuseExistingServer: true, env: {PORT: String(port)}},
  use: {
    baseURL: `http://127.0.0.1:${port}`,
    viewport: {width: 1440, height: 1050},
    launchOptions: {...(existsSync(chrome) ? {executablePath: chrome} : {}), args: ['--enable-unsafe-swiftshader']},
  },
});
