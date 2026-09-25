/// Fixed-height slot reserved for a future ad unit. Until then it shows a
/// looping ticker phrase. The height is fixed so the layout does not jump
/// when a real ad loads; swapping in the ad unit means replacing the ticker,
/// not the slot.
// 12 per half => 24 copies, ~5.8k px of track for this phrase. Comfortably
// wider than any window, so the ticker never runs out of text mid-window.
const REPEATS = 12;

export function AdBanner({
  text,
  speedSeconds = 18,
}: {
  text: string;
  speedSeconds?: number;
}) {
  // Two identical halves. The track slides by exactly half its width, so the
  // second half is always the first half's content arriving from the right:
  // the loop has no visible seam. REPEATS must be enough that one half
  // overfills the window, or the ticker would run out of text mid-window.
  const copies = Array.from({ length: REPEATS * 2 }, (_, i) => i);

  return (
    <aside
      aria-label="Advertisement"
      className="shrink-0 overflow-hidden border-t border-slate-200 bg-white dark:border-slate-800 dark:bg-slate-900"
      style={{ height: 100 }}
    >
      <div
        // ml-auto right-aligns the track, so the phrase starts flush against
        // the right edge and travels left instead of appearing mid-window.
        className="ml-auto flex h-full w-max items-center motion-safe:animate-ad-marquee"
        style={{ animationDuration: `${speedSeconds}s` }}
      >
        {copies.map((i) => (
          <span
            key={i}
            // Only the first copy is announced; the rest are visual repetition.
            aria-hidden={i > 0}
            className="px-8 text-sm font-medium whitespace-nowrap text-slate-500 dark:text-slate-400"
          >
            {text}
          </span>
        ))}
      </div>
    </aside>
  );
}
