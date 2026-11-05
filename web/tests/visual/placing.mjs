// The placing page checked in a browser: with no proposal it opens on points without sending
// the user away from them, and placing by points never saves the proposal the user set out to
// correct. Against `flaneur placer --no-live --answers <trial file>` running (the proposal laid
// in by the check, the answers counted as they are sent).
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
await page.route('**/api/answer', (route) => {
  answers.push(JSON.parse(route.request().postData()));
  return route.fulfill({ json: {} });
});
await page.route('**/api/composite', (route) => route.fulfill({ status: 204, body: '' }));

// no proposal: the page opens on points, and does not ask for the key that leaves them
await page.route('**/api/proposal/**', (route) => route.fulfill({ json: null }));
await page.goto(url);
check(await page.locator('#pair.on').waitFor({ timeout: 15000 }).then(() => true, () => false),
  'with no proposal, the page opens on points');
const told = await page.textContent('#message');
check(!told.includes('F pour'), `and does not send the user away from them (« ${told} »)`);
check(await page.locator('#fade').isDisabled(), 'and its slider says there is nothing to fade yet');
await page.unroute('**/api/proposal/**');

// every panorama offers a proposal (as matching would), on the photo shown
await page.route('**/api/proposal/**', (route) => {
  const panorama = decodeURIComponent(route.request().url().split('/').pop());
  return route.fulfill({ json: { registration: { ...registration, panorama }, carried_from: null } });
});
await page.goto(url);
await page.waitForTimeout(5000);
await page.keyboard.press('f'); // the proposal looks wrong: place it by points
await page.waitForTimeout(800);
const photo = await page.locator('#photo').boundingBox();
const view = await page.locator('#view').boundingBox();
const clickPairs = async (points) => {  // each point in the photo, then the same in the view
  for (const [x, y] of points) {
    await page.mouse.click(photo.x + x * photo.width, photo.y + y * photo.height);
    await page.mouse.click(view.x + x * view.width, view.y + y * view.height);
  }
};
await clickPairs([[0.1, 0.1], [0.9, 0.1], [0.5, 0.8]]);
await page.keyboard.press('Enter'); // three pairs only
await page.waitForTimeout(800);
check(answers.length === 0, `three pairs and Entrée save nothing (${answers.map((a) => a.answer)})`);
check((await page.textContent('#message')).includes('paires'), 'the page says pairs are missing');

// three more, six in all, that the server will not fit: Entrée gives its reason, not a count
// already reached
await page.route('**/api/fit', (route) => route.fulfill({ status: 400, body: 'points trop groupés' }));
await clickPairs([[0.1, 0.9], [0.9, 0.9], [0.5, 0.3]]);
await page.waitForTimeout(800);
await page.locator('#message').evaluate((m) => { m.textContent = ''; }); // as if read and gone
await page.keyboard.press('Enter');
await page.waitForTimeout(800);
const reason = await page.textContent('#message');
check(answers.length === 0 && reason.includes('groupés'), `six refused pairs and Entrée give the reason (« ${reason} »)`);

// a seventh pair that breaks the fit: the page says how to take it back
await clickPairs([[0.3, 0.6]]);
await page.waitForTimeout(800);
const back = await page.textContent('#message');
check(back.includes('⌫'), `a seventh pair refused says ⌫ takes it back (« ${back} »)`);
await browser.close();
process.exit(failed ? 1 : 0);
