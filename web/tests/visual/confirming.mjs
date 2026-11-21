// The confirmation page checked in a browser: a pair chosen is marked wrong with F, and taken
// back with F again; the mark shows on its loupe and both its points, is sent for this photo and
// that pair's point, and is never a verdict (the photo stays shown). Against `flaneur confirmer`
// running (its writes intercepted here: nothing is kept).
// Usage: node tests/visual/confirming.mjs [page-url]
import { chromium } from 'playwright';

const [url = 'http://127.0.0.1:8768/'] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1440, height: 900 } });
const sent = { pair: [], verdict: 0 };
let failed = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'ok ' : 'NOT'} ${what}`);
  if (!ok) failed++;
};
await page.route('**/api/pair', (route) => {
  sent.pair.push(route.request().postDataJSON());
  return route.fulfill({ json: {} });
});
await page.route('**/api/verdict', (route) => { sent.verdict++; return route.fulfill({ json: {} }); });
await page.goto(url);
await page.waitForFunction(() => photos.length > 0, null, { timeout: 60000 });
// the first photo whose pairs come
let found = false;
for (let i = 0; i < 20 && !found; i++) {
  found = await page.evaluate(async () => !!(await sceneOf(photos[at]))?.pairs?.length);
  if (!found) await page.keyboard.press('ArrowRight');
}
check(found, 'a photo with pairs is shown');
await page.keyboard.press('k');
await page.locator('.loupe').nth(1).waitFor({ timeout: 30000 });
const shownId = await page.evaluate(() => photos[at].id);
const index = await page.evaluate(async () => (await sceneOf(photos[at])).pairs[1].index);
await page.locator('.loupe').nth(1).click();
await page.keyboard.press('f');
await page.waitForFunction(() => document.querySelectorAll('.loupe.wrong').length === 1, null, { timeout: 5000 }).catch(() => {});
check(sent.pair.length === 1 && sent.pair[0].photo_id === shownId && sent.pair[0].index === index && sent.pair[0].wrong === true,
  `F marks the pair chosen wrong for this photo (${JSON.stringify(sent.pair[0])}, point ${index})`);
const look = await page.evaluate(() => [document.querySelectorAll('.loupe.wrong').length,
  document.querySelectorAll('#pairs .mark.wrong').length, document.querySelector('.loupe.wrong')?.dataset.n,
  document.getElementById('pairs-help').textContent, photos[at].id]);
check(look[0] === 1 && look[1] === 2 && look[2] === '1', `its loupe and both its points show it (${look.slice(0, 3)})`);
check(look[3].includes('1 marquée fausse'), `the line under the loupes counts it (« ${look[3]} »)`);
check(look[4] === shownId && sent.verdict === 0, 'the photo stays shown, no verdict sent');
check(await page.evaluate(() => chosen) === 1, 'the pair stays chosen');
await page.keyboard.press('f');
await page.waitForFunction(() => !document.querySelector('.loupe.wrong'), null, { timeout: 5000 }).catch(() => {});
check(sent.pair.length === 2 && sent.pair[1].wrong === false && !(await page.locator('.loupe.wrong').count()),
  'F again takes the mark back');
await page.keyboard.press('Escape');
await page.locator('.loupe').nth(1).click();  // chosen no more
await page.keyboard.press('f');
await page.waitForTimeout(300);
check(sent.pair.length === 2, 'no pair chosen: F sends nothing');
await browser.close();
process.exit(failed ? 1 : 0);
