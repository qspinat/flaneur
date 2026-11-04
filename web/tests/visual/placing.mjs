// The placing page checked in a browser: placing by points never saves the proposal the user
// set out to correct. Against `flaneur placer --no-live --answers <trial file>` running (the
// proposal laid in by the check, the answers counted as they are sent).
// Usage: node tests/visual/placing.mjs [page-url] <folder>/fixture.json (alignment_fixture.py's)
import { chromium } from 'playwright';
import { readFileSync } from 'node:fs';

const [url = 'http://127.0.0.1:8772/', fixture] = process.argv.slice(2);
const registration = JSON.parse(readFileSync(fixture, 'utf8')).registration;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const answers = [];
let failed = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'ok ' : 'NOT'} ${what}`);
  if (!ok) failed++;
};
// every panorama offers a proposal (as matching would), on the photo shown
await page.route('**/api/proposal/**', (route) => {
  const panorama = decodeURIComponent(route.request().url().split('/').pop());
  return route.fulfill({ json: { registration: { ...registration, panorama }, carried_from: null } });
});
await page.route('**/api/answer', (route) => {
  answers.push(JSON.parse(route.request().postData()));
  return route.fulfill({ json: {} });
});
await page.route('**/api/composite', (route) => route.fulfill({ status: 204, body: '' }));

await page.goto(url);
await page.waitForTimeout(5000);
await page.keyboard.press('f'); // the proposal looks wrong: place it by points
await page.waitForTimeout(800);
const photo = await page.locator('#photo').boundingBox();
const view = await page.locator('#view').boundingBox();
for (const [x, y] of [[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]) {
  await page.mouse.click(photo.x + x * photo.width, photo.y + y * photo.height);
  await page.mouse.click(view.x + x * view.width, view.y + y * view.height);
}
await page.keyboard.press('Enter'); // three pairs only
await page.waitForTimeout(800);
check(answers.length === 0, `three pairs and Entrée save nothing (${answers.map((a) => a.answer)})`);
check((await page.textContent('#message')).includes('paires'), 'the page says pairs are missing');
await browser.close();
process.exit(failed ? 1 : 0);
