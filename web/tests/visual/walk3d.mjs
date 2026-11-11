// The 3D walk checked in a browser, on the built site with its security policy: a photo laid in
// a panorama where the pipeline puts it (a checkerboard registered on a leaning panorama, see
// alignment_fixture.py: the page's view against the pipeline's drawing, within 2 px), no script
// error, nothing refused, the credits, a click on the photo opening its card, a step forward.
// Usage: node tests/visual/walk3d.mjs <base-url> <fixture-folder> [screenshot-folder]
import { chromium, firefox } from 'playwright';
import { readFileSync } from 'node:fs';
import { execFileSync } from 'node:child_process';

const [base, fixtures, out = fixtures] = process.argv.slice(2);
const fixture = JSON.parse(readFileSync(`${fixtures}/fixture.json`, 'utf8'));
const engine = process.env.BROWSER === 'firefox' ? firefox : chromium;
const browser = await engine.launch(engine === chromium ? { args: ['--use-angle=swiftshader', '--enable-unsafe-swiftshader'] } : {});
const page = await browser.newPage({ viewport: { width: 640, height: 640 } });
const errors = [];
page.on('pageerror', (e) => errors.push(e.message));
await page.addInitScript(() => {
  window.__refused = [];
  document.addEventListener('securitypolicyviolation', (e) => window.__refused.push(`${e.violatedDirective} ${e.blockedURI}`));
});
// the test photo, laid in the scene as the export would (nothing written in the site's data)
await page.route('**/data/scene.json', async (route) => {
  const scene = await (await route.fetch()).json();
  const home = scene.panoramas.find((p) => p.id === fixture.panorama);
  home.image = `panoramas/${fixture.panorama}.jpg`;
  scene.photos.push({ id: 'test:board', title: 'Damier', years: [1900, 1900], credit: 'Essai', image: 'photos/test-board.png', flipped: false, shown: [{ panorama: fixture.panorama, registration: fixture.registration, strength: 1 }] });
  await route.fulfill({ json: scene });
});
await page.route(`**/data/panoramas/${fixture.panorama}.jpg`, (route) => route.fulfill({ path: `${fixtures}/panorama.jpg` }));
await page.route('**/data/photos/test-board.png', (route) => route.fulfill({ path: `${fixtures}/board.png` }));

let failed = 0;
const check = (ok, what) => {
  console.log(`${ok ? 'ok ' : 'NOT'} ${what}`);
  if (!ok) failed++;
};

await page.goto(`${base}3d.html#photo=test%3Aboard,1850-2026`); // as the photo walk's "En 3D" opens it
// the policy blocks waitForFunction's evaluation: evaluate is not, polled
const ready = async () => {
  for (let i = 0; i < 100 && !(await page.evaluate(() => window.ready)); i++) await page.waitForTimeout(100);
  await page.waitForLoadState('networkidle'); // the photo's image too
};
await ready();
await page.locator('#fade').fill('100'); // the photo at full strength, as the pipeline draws it
await page.evaluate(() => document.querySelectorAll('body > :not(#view)').forEach((e) => (e.style.visibility = 'hidden')));
await page.waitForTimeout(800);
await page.screenshot({ path: `${out}/walk3d-shown.png` });
const python = process.env.PYTHON ?? '../pipeline/.venv/bin/python'; // the pipeline's, with OpenCV
const measured = execFileSync(python, ['-c', `
import cv2, numpy as np, sys
red = lambda im: ((im[:,:,2] > 200) & (im[:,:,0] < 90) & (im[:,:,1] < 90)).astype(np.float32)
a, b = red(cv2.imread(sys.argv[1])), red(cv2.imread(sys.argv[2]))
(dx, dy), _ = cv2.phaseCorrelate(a, b)
print(f"{((a > 0) & (b > 0)).sum() / max(1, ((a > 0) | (b > 0)).sum()):.3f} {abs(dx):.2f} {abs(dy):.2f} {int(b.sum())}")
`, `${fixtures}/expected.png`, `${out}/walk3d-shown.png`]).toString().trim().split(' ').map(Number);
const [overlap, dx, dy, shown] = measured;
console.log(`alignment: overlap ${overlap}, shift ${dx} px, ${dy} px (${shown} red pixels shown)`);
// a shift means something only where the boards overlap
check(shown > 1000 && overlap >= 0.9 && Math.hypot(dx, dy) <= 2, 'the photo lies where the pipeline puts it (within 2 px)');

await page.evaluate(() => document.querySelectorAll('body > *').forEach((e) => (e.style.visibility = '')));
const credits = await page.locator('#credits-list').textContent(); // folded away: its text still
check(credits.includes("Rue d'aujourd'hui") && credits.includes('Essai'), 'today and the photo are credited');
await page.mouse.click(320, 300); // on the board
await page.waitForTimeout(500);
check(await page.locator('#card').isVisible(), 'a click on the photo opens its card');
const before = await page.evaluate(() => location.hash);
await page.keyboard.press('Escape');
await page.evaluate(() => (window.ready = false));
await page.keyboard.press('ArrowUp');
await ready();
await page.waitForTimeout(500); // the link is written once the view rests
check((await page.evaluate(() => location.hash)) !== before, 'a step forward goes to the next panorama');
// a panorama that cannot be loaded: one stays where one stood, told so
await page.route(/panoramax|\/data\/panoramas\//, (route) => route.abort());
await page.evaluate(() => (window.ready = false));
const stood = await page.evaluate(() => location.hash);
await page.evaluate((f) => (location.hash = `#${(f.lon + 0.002).toFixed(5)},${f.lat.toFixed(5)},0,1850-2026`), fixture); // a panorama 150 m east
await ready();
await page.waitForTimeout(500);
await page.waitForTimeout(500); // the link written again where one stood
check((await page.locator('#notice').isVisible()) && (await page.evaluate(() => location.hash)) === stood, 'a view that cannot be loaded leaves one where one stood, told so');
check(!(await page.evaluate(() => window.__refused)).length, `nothing refused by the policy ${JSON.stringify(await page.evaluate(() => window.__refused))}`);
check(!errors.length, `no script error ${errors.join(' | ')}`);
await browser.close();
process.exit(failed ? 1 : 0);
