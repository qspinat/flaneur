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
  const click = async (points) => {  // where the pictures stand now (a third one moves them)
    const shot = await opened.locator('#photo').boundingBox(), seen = await opened.locator('#view').boundingBox();
    for (const [x, y] of points) {
      await opened.mouse.click(shot.x + x * shot.width, shot.y + y * shot.height);
      await opened.mouse.click(seen.x + x * seen.width, seen.y + y * seen.height);
    }
  };
  return [opened, click];
}

const SIX = [[0.2, 0.2], [0.8, 0.2], [0.5, 0.7], [0.3, 0.5], [0.7, 0.6], [0.4, 0.3]];
const JPEG = { contentType: 'image/jpeg', body: Buffer.from([0xff, 0xd8, 0xff, 0xd9]) };  // a picture
const CENTRE = [2.3459, 48.8518];  // the map's, when a test needs a spot
/** Walls around a spot, as /api/depth sends them: one fitted, one drawn here, BD TOPO's facade. */
const wallsAround = (lon, lat, panorama, neighbour) => {
  const at = (east, north) => [lon + east / (111320 * Math.cos(lat * Math.PI / 180)), lat + north / 111320];
  return { details: 900, neighbours: [neighbour], best: neighbour, why: '',
    candidates: [{ id: neighbour, metres: 4.2, fixed: true }],
    walls: [{ a: at(-6, 0), b: at(-6, 30), top: 12, made: 'fitted', at: '', mine: null },
            { a: at(6, 0), b: at(6, 30), top: 20, made: 'drawn', at: panorama, mine: 0 }],
    segments: [[at(-12, -5), at(-12, 35)]],
    mine: { drawn: [{ a: at(6, 0), b: at(6, 30), top: null }], removed: [] } };
};
const asked = { depth: [], parallax: [], standingLaid: 0, answer: null, epipolar: 0, second: null, walls: [], tint: 0 };
let stood = null, here = null;
const walls = (record) => (route) => {
  const body = route.request().postDataJSON();
  record.push(body);
  return route.fulfill({ json: wallsAround(here.lon, here.lat, body.panorama, here.neighbour) });
};
const [deep, pairsAt] = await onPoints({
  '**/api/depth': walls(asked.depth),
  '**/api/walls': walls(asked.walls),
  '**/api/parallax': (route) => {
    const body = route.request().postDataJSON();
    asked.parallax.push(body);
    stood = { lon: body.lon + 0.00004, lat: body.lat - 0.00002 };
    const n = body.photo_px.length;
    return route.fulfill({ json: {
      registration: { ...registration, panorama: body.panorama }, ...stood, height_above_ground: 1.6, heading: 120,
      standing: { position: [0, 0, 0], turn: [0, 0, 0], focal: 500, shift: [0, 0], size: [600, 800] },
      used: Array(n).fill(true), metres: Array(n).fill(12), source: Array(n).fill('wall'),
      neighbours: [here.neighbour], chosen: null, errors: [0.005, 0.02, 0.05, ...Array(n).fill(0.001)].slice(0, n),
      held_out: { flat: Array(n).fill(0.04), parallax: Array(n).fill(0.01) } } });
  },
  '**/api/epipolar': (route) => { asked.epipolar++; return route.fulfill({ json: { line: [[100, 600], [700, 640], [1200, 700]] } }); },
  '**/api/second': (route) => { asked.second = route.request().postDataJSON(); return route.fulfill({ json: { metres: 14.2 } }); },
  '**/api/depthview': (route) => { asked.tint++; return route.fulfill(JPEG); },
  '**/api/wallsview': (route) => route.fulfill({ json: { walls: [
    { i: 0, made: 'fitted', foot: [[100, 900], [500, 800]], top: [[100, 300], [500, 400]] }] } }),
  '**/api/composite': (route) => {
    if (route.request().postDataJSON().standing) asked.standingLaid++;
    return route.fulfill(JPEG);
  },
  '**/api/answer': (route) => { asked.answer = route.request().postDataJSON(); return route.fulfill({ json: {} }); },
});
here = await deep.evaluate(() => ({ lon: near[at].lon, lat: near[at].lat, id: near[at].id,
                                     neighbour: (near[1] ?? near[0]).id, at: [(near[1] ?? near[0]).lon, (near[1] ?? near[0]).lat] }));
/** Where a point ([lon, lat]) of the map is on the screen. */
const onScreen = (point) => deep.evaluate((p) => {
  const at = map.project(p), box = map.getCanvas().getBoundingClientRect();
  return [box.left + at.x, box.top + at.y];
}, point);
const clickMap = async (point, key) => {
  const [x, y] = await onScreen(point);
  if (key) await deep.keyboard.down(key);
  await deep.mouse.click(x, y);
  if (key) await deep.keyboard.up(key);
  await deep.waitForTimeout(700);
};
await deep.keyboard.press('p');
await deep.waitForTimeout(800);
check(asked.depth.length === 1, `the toggle makes the panorama's walls ahead (${asked.depth.length})`);
const drawn = await deep.evaluate(() => [map.getSource('walls').serialize().data.features.length,
                                         map.getSource('segments').serialize().data.features.length]);
check(drawn[0] === 2 && drawn[1] === 1, `the walls and BD TOPO's facades are drawn on the map (${drawn})`);
await deep.waitForTimeout(300);
const overlaid = await deep.locator('#view-stage .in-view polyline').count();
check(overlaid >= 3, `the walls are drawn over the view too: foot, top, ends (${overlaid})`);
check(!(await deep.locator('#third-box').isVisible()), 'pairing keeps two large pictures: the photo and the view');
await pairsAt(SIX);
await deep.waitForTimeout(1000);
check(asked.parallax.length > 0 && asked.standingLaid > 0, `six pairs refit by parallax and laid through depth (${asked.parallax.length}, ${asked.standingLaid})`);
check((await deep.locator('#view-stage .dot.off').count()) >= 1, 'a pair far from the fit shows on its dot');
check((await deep.locator('#strip button.wall').count()) === 6, 'each pair\'s plate says its depth came from a wall');
await deep.locator('#strip button').nth(5).click();  // pair 6: its depth by the neighbour's view
await deep.waitForTimeout(500);
check(await deep.locator('#third-box').isVisible() && !(await deep.locator('#photo-box').isVisible()),
  'a pair\'s plate shows the neighbour\'s view in the photo\'s place');
check(asked.epipolar > 0 && (await deep.locator('#line').getAttribute('points')).length > 0,
  `the pair's line is drawn in the neighbour's view (${asked.epipolar})`);
const third = await deep.locator('#third').boundingBox();
await deep.mouse.click(third.x + third.width * 0.5, third.y + third.height * 0.5);
await deep.waitForTimeout(1000);
check(asked.second?.neighbour === here.neighbour && asked.second?.neighbour_px?.length === 2,
  'a click in the neighbour\'s view asks the last pair\'s depth there');
check((await deep.locator('#view-stage .dot.click').count()) === 1, 'that pair\'s dot says its depth came from a second click');
check(await deep.locator('#photo-box').isVisible(), 'and the photo comes back');
check(asked.parallax.at(-1).second_clicks?.[5]?.neighbour === here.neighbour, 'the parallax fit is made again with the second click');
await deep.locator('#tab-neighbours').click();
check(await deep.locator(`#nb-${here.neighbour}`).isChecked(), 'the neighbour used is ticked in the list');
await deep.locator(`#nb-${here.neighbour}`).uncheck();
await deep.waitForTimeout(700);
check(Array.isArray(asked.depth.at(-1).neighbours) && !asked.depth.at(-1).neighbours.includes(here.neighbour),
  `a neighbour unticked is left out of the walls (${JSON.stringify(asked.depth.at(-1).neighbours)})`);
await deep.locator('#automatic').click();
await deep.waitForTimeout(700);
check(asked.depth.at(-1).neighbours === undefined, 'the automatic choice comes back with its button');
await deep.locator('#tab-walls').click();
check((await deep.textContent('#mapmode')).includes('choisir'), 'the walls tool says a click chooses a wall');
await deep.locator('#draw-walls').click();
const metres = (east, north) => [here.lon + east / (111320 * Math.cos(here.lat * Math.PI / 180)), here.lat + north / 111320];
await clickMap(metres(-15, -10)); await clickMap(metres(-15, 10));
check(asked.walls.at(-1)?.drawn?.length === 2 && asked.walls.at(-1).removed.length === 0,
  `a wall drawn in two clicks on the map is saved with the panorama's own (${asked.walls.at(-1)?.drawn?.length})`);
await deep.locator('#draw-walls').click();  // drawing done: a click chooses again
const before = asked.walls.length;
await clickMap(metres(-12, 25));  // BD TOPO's facade, grey, away from the walls
check(asked.walls.length === before + 1 && asked.walls.at(-1).drawn.length === 2,
  'a BD TOPO facade clicked is saved as a wall drawn here');
await clickMap(metres(-6, 15));  // the fitted wall
check(await deep.locator('#chosen-wall').isVisible(), 'a wall chosen shows what it is and how to remove it');
await deep.locator('#remove-wall').click();
await deep.waitForTimeout(700);
check(asked.walls.at(-1)?.removed?.length === 1, `a fitted wall chosen and ⌫ is saved as removed here (${asked.walls.at(-1)?.removed?.length})`);
await clickMap(metres(6, 15));  // the wall drawn here
const end = await deep.locator('.wall-end').first().boundingBox();
await deep.mouse.move(end.x + end.width / 2, end.y + end.height / 2);
await deep.mouse.down(); await deep.mouse.move(end.x + 40, end.y + 10, { steps: 5 }); await deep.mouse.up();
await deep.waitForTimeout(700);
const moved = asked.walls.at(-1);
check(moved?.drawn?.length === 1 && moved.drawn[0].a[0] !== wallsAround(here.lon, here.lat, '', '').mine.drawn[0].a[0],
  'dragging an end of a wall drawn here moves it');
await deep.locator('#tab-walls').click();  // the tool put away
await deep.locator('#tint').check();
await deep.waitForTimeout(600);
check(asked.tint > 0, 'the tint box shows the view tinted by depth');
const plates = await deep.locator('#strip button').count();
check(plates === 6, `the strip shows the six pairs (${plates})`);
await deep.locator('#strip button').nth(2).click();
check((await deep.textContent('#third-label')).includes('Paire 3'), 'a plate clicked aims the next click beside at its pair');
await deep.keyboard.press('Escape');
check(await deep.locator('#photo-box').isVisible(), 'Échap brings the photo back');
await deep.locator('#tint').uncheck();
await deep.waitForTimeout(1500);  // the view itself again
await deep.screenshot({ path: 'screenshots/placing-walls.png' });
check(await deep.locator('#stood-ok').isVisible(), 'a button offers to put the camera where the fit found it');
await deep.locator('#stood-ok').click();
await deep.waitForTimeout(300);
const pin = await deep.evaluate(() => spot);
check(Math.abs(pin.lon - stood.lon) < 1e-9 && Math.abs(pin.lat - stood.lat) < 1e-9 && pin.heading === 120,
  'clicking where the photographer stood moves the pin there, the cone turned as its camera looked');
await deep.keyboard.press('Enter');
await deep.waitForTimeout(800);
const sent = asked.answer || {};
check(sent.parallax && sent.parallax.lon !== stood.lon && sent.pairs?.photo_px?.length === 6
  && sent.pairs?.second_clicks?.[5]?.neighbour === here.neighbour,
  `Entrée keeps the pairs, their second clicks and the pin the parallax fit started from (${JSON.stringify(sent.parallax)})`);

// parallax refused (too few pairs with depth): the flat fit meanwhile, so the photo can be kept
const [flatPage, flatPairs] = await onPoints({
  '**/api/depth': (route) => route.fulfill({ json: { ...wallsAround(...CENTRE, 'x', 'y'), neighbours: [] } }),
  '**/api/parallax': (route) => route.fulfill({ status: 400, body: 'il faut au moins 6 paires avec une profondeur' }),
});
await flatPage.keyboard.press('p');
await flatPage.waitForTimeout(300);
await flatPairs(SIX);
await flatPage.waitForTimeout(1500);
const flatSaid = await flatPage.textContent('#message');
check(flatSaid.includes('à plat') && (await flatPage.evaluate(() => !!proposal)), `parallax refused falls back to the flat fit (« ${flatSaid} »)`);

// a panorama COLMAP did not place: the page says so, and offers the nearest one placed
const [unsure] = await onPoints({
  '**/api/near**': async (route) => {
    const found = await (await route.fetch()).json();
    return route.fulfill({ json: found.map((p, i) => ({ ...p, fixed: i > 0 })) });
  },
  '**/api/depth': (route) => route.fulfill({ json: wallsAround(...CENTRE, 'x', 'y') }),
});
await unsure.keyboard.press('p');
await unsure.waitForTimeout(600);
const doubt = await unsure.textContent('#message');
check(doubt.includes('COLMAP'), `a panorama COLMAP did not place says its depth is unsure (« ${doubt} »)`);

// walls asked at one panorama, coming after the page went to the next: not taken for its own
let slow = true;
const [late] = await onPoints({
  '**/api/depth': async (route) => {
    const first = slow;
    slow = false;
    if (first) await new Promise((done) => setTimeout(done, 2500));
    return route.fulfill({ json: { ...wallsAround(...CENTRE, 'x', 'y'), details: first ? 111 : 222 } });
  },
});
await late.keyboard.press('p');  // the walls asked here: slow
await late.waitForTimeout(300);
await late.keyboard.press('p');  // parallax off, and on to the next panorama while they come
await late.keyboard.press('.');
await late.waitForTimeout(3500);
const kept = await late.evaluate(() => scene?.details);
check(kept !== 111, `walls coming late from the panorama left are not taken for the one shown (${kept})`);

// no depth to be had: the toggle says why
const [none] = await onPoints({
  '**/api/depth': (route) => route.fulfill({ status: 400, body: 'pas de panorama voisin à 2–15 m' }),
});
await none.keyboard.press('p');
await none.waitForTimeout(500);
const button = none.locator('#parallax');
check(await button.isDisabled() && (await button.getAttribute('title')).includes('voisin'), 'without depth, the toggle says why');
await browser.close();
process.exit(failed ? 1 : 0);
