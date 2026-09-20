/**
 * VERIFICATION TEST — the tick clock
 * ===========================================================================
 * Every tick counts against real time, wherever it came from:
 *
 *   1. RATE      — left to the timer, the brain takes the configured ticks
 *                  per second, no more.
 *   2. LEAD      — ticks run ahead of time (a perception propagated at
 *                  once) are waited out: the timer runs nothing until real
 *                  time has caught up, and the total stays at the rate.
 *   3. LAG       — after a stall the timer catches up in bounded steps.
 *   4. FORGIVEN  — a lead or a lag beyond the limits is forgiven: a long
 *                  lesson does not freeze the brain for minutes, a laptop
 *                  asleep for an hour does not wake to an hour of ticks.
 */
import { TickClock } from '../src/tick-clock.js';

const results: Array<[string, boolean]> = [];
const check = (name: string, ok: boolean, detail = ''): void => {
  results.push([name, ok]);
  console.log(`   ${ok ? '✅' : '❌'} ${name}${detail ? `  (${detail})` : ''}`);
};

console.log('── Verification: the tick clock ──\n');
const INTERVAL = 100; // 10 Hz
const make = (speed = 1): TickClock => new TickClock({ intervalMs: INTERVAL, speed, maxLeadTicks: 100, maxLagTicks: 600, maxCatchUp: 3 * speed, now: 0 });
/** Runs the timer's callback every interval from `from` to `to` ms, returning the ticks it ran. */
const runTimer = (clock: TickClock, from: number, to: number): number => {
  let ran = 0;
  for (let t = from; t <= to; t += INTERVAL) { const n = clock.due(t); ran += n; clock.credit(n); }
  return ran;
};

console.log('1. RATE');
{
  const clock = make();
  const ran = runTimer(clock, INTERVAL, 10_000);
  check('ten seconds of timer are a hundred ticks at 10 Hz', ran === 100, `${ran} ticks`);
  const fast = make(5);
  const ranFast = runTimer(fast, INTERVAL, 10_000);
  check('…and five hundred at speed 5', ranFast === 500, `${ranFast} ticks`);
}

console.log('\n2. LEAD');
{
  const clock = make();
  runTimer(clock, INTERVAL, 1000); // 10 ticks in the first second
  clock.credit(50); // a perception propagated at once: 50 ticks ahead of time
  check('the brain is now ahead of real time', clock.lead(1000) === 50, `lead ${clock.lead(1000)} ticks`);
  const during = runTimer(clock, 1100, 6000); // the next five seconds
  check('the timer waits the lead out', during === 0, `${during} timer ticks while ahead`);
  const after = runTimer(clock, 6100, 10_000);
  check('and the total over ten seconds is still a hundred', 10 + 50 + during + after === 100, `${10 + 50 + during + after} ticks`);
}

console.log('\n3. LAG');
{
  const clock = make();
  runTimer(clock, INTERVAL, 1000);
  // A stall: the timer does not fire for two seconds.
  const first = clock.due(3000);
  check('after a stall the timer catches up in bounded steps', first === 3, `${first} ticks at the first fire after the stall`);
  const caught = runTimer(clock, 3000, 4000);
  check('…and is level within a second', caught === 30 && clock.lead(4000) === 0, `${caught} ticks over the next second, lead ${clock.lead(4000)}`);
}

console.log('\n4. FORGIVEN');
{
  const clock = make();
  clock.credit(5000); // a long lesson: five thousand ticks at once
  const lead = clock.lead(0);
  clock.due(0);
  check('a lead beyond the limit is forgiven', lead === 5000 && clock.lead(0) === 100, `lead ${lead} → ${clock.lead(0)} ticks`);
  const slept = make();
  const n = slept.due(3_600_000); // an hour later
  check('an hour asleep does not wake to an hour of ticks', n === 3 && slept.lead(3_600_000) >= -600, `${n} ticks due, lag ${-slept.lead(3_600_000)} ticks`);
}

const failed = results.filter(([, ok]) => !ok);
console.log(`\n${results.length - failed.length}/${results.length} checks passed`);
if (failed.length > 0) {
  console.log('Failed:\n' + failed.map(([name]) => `   - ${name}`).join('\n'));
  process.exit(1);
}
