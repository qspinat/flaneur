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
await clickPairs([[0.1, 0.1]]);
await page.keyboard.press('Enter'); // one pair only
await page.waitForTimeout(800);
check(answers.length === 0, `one pair and Entrée save nothing (${answers.map((a) => a.answer)})`);
check((await page.textContent('#message')).includes('paires'), 'the page says pairs are missing');

// zoomed in, a pair's dots keep their size on screen
await page.mouse.move(photo.x + photo.width / 2, photo.y + photo.height / 2);
await page.mouse.wheel(0, -600);
await page.waitForTimeout(300);
const dot = await page.locator('#photo-stage .dot').first().boundingBox();
check(Math.abs(dot.width - 20) < 1, `zoomed in, a dot keeps its size (${dot.width.toFixed(1)} px)`);
await page.locator('#photo-box').dispatchEvent('dblclick'); // back to the whole photo, no click on it
await page.waitForTimeout(300);

// two more that the server will not fit: Entrée gives its reason, not a count already reached
await page.route('**/api/fit', (route) => route.fulfill({ status: 400, body: 'points trop groupés' }));
await clickPairs([[0.9, 0.1], [0.5, 0.8]]);
await page.waitForTimeout(800);
await page.locator('#message').evaluate((m) => { m.textContent = ''; }); // as if read and gone
await page.keyboard.press('Enter');
await page.waitForTimeout(800);
const reason = await page.textContent('#message');
check(answers.length === 0 && reason.includes('groupés'), `refused pairs and Entrée give the reason (« ${reason} »)`);

// one pair more that breaks the fit: the page says how to take it back
await clickPairs([[0.3, 0.6]]);
await page.waitForTimeout(800);
const back = await page.textContent('#message');
check(back.includes('⌫'), `a pair more refused says ⌫ takes it back (« ${back} »)`);

// parallax: the toggle refits the pairs by where the photographer stood
/** A page opened on points (no proposal), and a way to click pairs in it. */
async function onPoints(routes) {
  const opened = await browser.newPage({ viewport: { width: 1440, height: 900 } });
  await opened.route('**/api/proposal/**', (route) => route.fulfill({ json: null }));
  for (const [path, handle] of Object.entries(routes)) await opened.route(path, handle);
  await opened.goto(url);
  await opened.locator('#pair.on').waitFor({ timeout: 15000 });
  await opened.waitForTimeout(1000);
  const shot = await opened.locator('#photo').boundingBox(), seen = await opened.locator('#view').boundingBox();
  const click = async (points) => {
    for (const [x, y] of points) {
      await opened.mouse.click(shot.x + x * shot.width, shot.y + y * shot.height);
      await opened.mouse.click(seen.x + x * seen.width, seen.y + y * seen.height);
    }
  };
  return [opened, click];
}

const asked = { depth: 0, parallax: 0, standingLaid: 0, answer: null };
let stood = null;
const [deep, pairsAt] = await onPoints({
  '**/api/depth': (route) => { asked.depth++; return route.fulfill({ json: { details: 900 } }); },
  '**/api/parallax': (route) => {
    asked.parallax++;
    const body = JSON.parse(route.request().postData());
    stood = { lon: body.lon + 0.00004, lat: body.lat - 0.00002 };
    const n = body.photo_px.length;
    return route.fulfill({ json: {
      registration: { ...registration, panorama: body.panorama }, ...stood, height_above_ground: 1.6, heading: 120,
      standing: { position: [0, 0, 0], turn: [0, 0, 0], focal: 500, shift: [0, 0], size: [600, 800] },
      used: Array(n).fill(true), metres: Array(n).fill(12), source: Array(n).fill('wall'),
      neighbours: ['p2'], errors: [0.005, 0.02, 0.05, ...Array(n).fill(0.001)].slice(0, n),
      held_out: { flat: Array(n).fill(0.04), parallax: Array(n).fill(0.01) } } });
  },
  '**/api/composite': (route) => {
    if (JSON.parse(route.request().postData()).standing) asked.standingLaid++;
    return route.fulfill({ contentType: 'image/jpeg', body: Buffer.from([0xff, 0xd8, 0xff, 0xd9]) });
  },
  '**/api/answer': (route) => { asked.answer = JSON.parse(route.request().postData()); return route.fulfill({ json: {} }); },
});
await deep.keyboard.press('p');
await deep.waitForTimeout(500);
check(asked.depth === 1, `the toggle makes the view's depth ahead (${asked.depth})`);
await pairsAt([[0.2, 0.2], [0.8, 0.2], [0.5, 0.7], [0.3, 0.5]]);
await deep.waitForTimeout(800);
check(asked.parallax > 0 && asked.standingLaid > 0, `four pairs refit by parallax and laid through depth (${asked.parallax}, ${asked.standingLaid})`);
check((await deep.locator('#view-stage .dot.off').count()) >= 1, 'a pair far from the fit shows on its dot');
await deep.locator('.stood').click();
await deep.waitForTimeout(300);
const pin = await deep.evaluate(() => spot);
check(Math.abs(pin.lon - stood.lon) < 1e-9 && Math.abs(pin.lat - stood.lat) < 1e-9 && pin.heading === 120,
  'clicking where the photographer stood moves the pin there, the cone turned as its camera looked');
await deep.keyboard.press('Enter');
await deep.waitForTimeout(800);
const sent = asked.answer || {};
check(sent.parallax && sent.parallax.lon !== stood.lon && sent.pairs?.photo_px?.length === 4,
  `Entrée keeps the pairs and the pin the parallax fit started from (${JSON.stringify(sent.parallax)})`);

// parallax refused (too few pairs with depth): the flat fit meanwhile, so the photo can be kept
const [flatPage, flatPairs] = await onPoints({
  '**/api/depth': (route) => route.fulfill({ json: { details: 900 } }),
  '**/api/parallax': (route) => route.fulfill({ status: 400, body: 'il faut au moins 4 paires avec une profondeur' }),
});
await flatPage.keyboard.press('p');
await flatPage.waitForTimeout(300);
await flatPairs([[0.2, 0.2], [0.8, 0.2], [0.5, 0.7], [0.3, 0.5]]);
await flatPage.waitForTimeout(1500);
const flatSaid = await flatPage.textContent('#message');
check(flatSaid.includes('à plat') && (await flatPage.evaluate(() => !!proposal)), `parallax refused falls back to the flat fit (« ${flatSaid} »)`);

// no depth to be had: the toggle says why
const [none] = await onPoints({
  '**/api/depth': (route) => route.fulfill({ status: 400, body: 'pas de panorama voisin à 1–12 m' }),
});
await none.keyboard.press('p');
await none.waitForTimeout(500);
const button = none.locator('#parallax');
check(await button.isDisabled() && (await button.getAttribute('title')).includes('voisin'), 'without depth, the toggle says why');
await browser.close();
process.exit(failed ? 1 : 0);
