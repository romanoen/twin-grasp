// Browser tests of the station viewer. Station state, camera image and live arm packet are mocked; the arm snapshot
// and the work area come from the real data files.
import {test, expect} from '@playwright/test';
import {readFileSync} from 'node:fs';
const snapshot = JSON.parse(readFileSync(new URL('../data/robot.json', import.meta.url)));
const jpg = readFileSync(new URL('../../tests/fixtures/grasp_20261109_101521/locate.jpg', import.meta.url));
const state = (over = {}) => ({timestamp_ms: Date.now(), phase: 'ready', message: 'Cube in map - ready to grasp', held: true,
  holding_cube: false, cube: {q: [40, 200, 15], state: 'locked'}, can_locate: true, can_grasp: true, can_place: false, ...over});

async function open(page, getState, sent = []) {
  await page.route('**/data/grasp-state.json', route => route.fulfill({json: {...getState(), timestamp_ms: Date.now()}}));
  await page.route('**/data/camera.jpg*', route => route.fulfill({body: jpg, contentType: 'image/jpeg'}));
  await page.route('**/api/command', route => { sent.push(route.request().postDataJSON()); route.fulfill({json: {ok: true}}); });
  await page.addInitScript(() => localStorage.clear());
  await page.goto('/'); await page.waitForFunction(() => window.station?.state());
  await page.waitForFunction(() => window.station.robotParts() > 0);
}

test('layout: 3D view, arm, work area, camera, cube and five commands', async ({page}) => {
  const sent = [];
  await open(page, () => state(), sent);
  expect(await page.evaluate(() => window.station.robotParts())).toBe(17);
  expect(await page.evaluate(() => window.station.areaCells())).toBeGreaterThan(100);
  for (const [id, label] of [['stop', 'STOP'], ['locate', 'LOCATE'], ['home', 'RESET'], ['grasp', 'GRASP'], ['place', 'PLACE']]) await expect(page.locator(`#cmd-${id}`)).toContainText(label);
  await expect(page.locator('#phase')).toHaveText('READY');
  await expect(page.locator('#log li').first()).toContainText('Cube in map');
  await expect(page.locator('#message')).toHaveText('Cube in map - ready to grasp');
  await expect(page.locator('#camera')).toHaveAttribute('src', /camera\.jpg/);
  await expect(page.locator('#cube-x')).toHaveText('40.0');
  const cube = await page.evaluate(() => window.station.cube());
  expect(cube.visible).toBe(true); expect(cube.position[0]).toBeCloseTo(.04, 5);
  await expect(page.locator('#cmd-place')).toBeDisabled();
  for (const id of ['stop', 'locate', 'home', 'grasp']) await page.locator(`#cmd-${id}`).click();
  await page.keyboard.press('Escape'); await page.keyboard.press('l');
  await expect.poll(() => sent.map(s => s.cmd)).toEqual(['stop', 'locate', 'home', 'grasp', 'stop', 'locate']);
  await page.screenshot({path: 'test-results/station.png'});
});

test('place sends the target; outside the work area it is blocked', async ({page}) => {
  const sent = [];
  await open(page, () => state({cube: {q: [10, 180, 0], state: 'held', z_mm: 60}, holding_cube: true, can_grasp: false, can_place: true, can_locate: false}), sent);
  await page.locator('#target-x').fill('30'); await page.locator('#target-x').press('Enter');
  await page.locator('#target-yaw').fill('25'); await page.locator('#target-yaw').press('Enter');
  expect(await page.evaluate(() => window.station.target())).toEqual({xy_mm: [30, 230], yaw_deg: 25});
  await expect(page.locator('#cmd-place')).toBeEnabled();
  await page.locator('#cmd-place').click();
  await expect.poll(() => sent.at(-1)).toEqual({cmd: 'place', x_mm: 30, y_mm: 230, yaw_deg: 25});
  await page.locator('#target-x').fill('380'); await page.locator('#target-x').press('Enter');
  await expect(page.locator('#cmd-place')).toBeDisabled();
  await expect(page.locator('#hint')).toContainText('outside');
});

test('target can be dragged and rotated with Q/E', async ({page}) => {
  await open(page, () => state());
  const before = await page.evaluate(() => window.station.target());
  const p = await page.evaluate(() => window.station.targetScreenPoint());
  await page.mouse.move(p.x, p.y); await page.mouse.down(); await page.mouse.move(p.x + 60, p.y + 10, {steps: 5}); await page.mouse.up();
  const after = await page.evaluate(() => window.station.target());
  expect(Math.hypot(after.xy_mm[0] - before.xy_mm[0], after.xy_mm[1] - before.xy_mm[1])).toBeGreaterThan(5);
  await page.locator('#viewport canvas').focus(); await page.keyboard.press('q');
  expect((await page.evaluate(() => window.station.target())).yaw_deg).toBe(5);
});

test('offline: commands disabled, no camera image', async ({page}) => {
  await page.route('**/data/grasp-state.json', route => route.fulfill({status: 404, body: ''}));
  await page.goto('/'); await page.waitForFunction(() => window.station);
  await expect(page.locator('#phase')).toHaveText('OFF');
  for (const id of ['stop', 'locate', 'home', 'grasp', 'place']) await expect(page.locator(`#cmd-${id}`)).toBeDisabled();
  await expect(page.locator('#camera')).not.toHaveAttribute('src', /.+/);
  expect((await page.evaluate(() => window.station.cube())).visible).toBe(false);
});

test('live arm pose is applied; mismatched model is ignored', async ({page}) => {
  // live packet = the static snapshot with a fresh timestamp (no hardware needed)
  await page.route('**/data/live-robot.json', route => route.fulfill({json: {...snapshot, timestamp_ms: Date.now()}}));
  await open(page, () => state());
  await expect.poll(() => page.evaluate(() => window.station.armLive())).toBe(true);
  await page.route('**/data/live-robot.json', route => route.fulfill({json: {model_id: 'other', timestamp_ms: Date.now()}}));
  await expect.poll(() => page.evaluate(() => window.station.armLive())).toBe(false);
});

test('busy phase: only STOP is available', async ({page}) => {
  await open(page, () => state({phase: 'grasping', message: 'Descending ...'}));
  await expect(page.locator('#cmd-stop')).toBeEnabled();
  for (const id of ['locate', 'home', 'grasp', 'place']) await expect(page.locator(`#cmd-${id}`)).toBeDisabled();
  await expect(page.locator('#status')).toHaveAttribute('data-tone', 'busy');
});

test('phone width: no horizontal scroll', async ({page}) => {
  await page.setViewportSize({width: 390, height: 844});
  await open(page, () => state());
  expect(await page.evaluate(() => document.documentElement.scrollWidth)).toBeLessThanOrEqual(390);
});
