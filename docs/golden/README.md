# Golden transcripts

These files are ground truth: they are what **openai/whisper itself** produces
for `test_16000_mono.wav` (22.1 s of Hebrew speech at 16 kHz mono). They exist
to answer one question — does our Rust port behave like the original, or does
it merely agree with itself?

## They are generated, not hand-written

Never edit these by hand. Regenerate them with:

```text
python scripts/cement_reference.py --model large-v3 --language auto
```

The reference implementation is **not** a dependency of this repository. You
need it separately (`pip install openai-whisper torch`) on a Python that torch
publishes wheels for — cp311 is safe, cp314 is not.

Every golden is accompanied by a `.provenance.json` recording the whisper and
torch versions, the search settings, the audio and the resulting segment
count. A reference file without recorded provenance cannot be reproduced, so it
cannot be trusted or deliberately invalidated — it just becomes folklore.

## The settings matter

The goldens are generated with `beam_size=5` and `temperature=0.0`.
`temperature=0.0` disables openai/whisper's temperature-fallback chain, which
re-decodes on low confidence and is the single largest source of run-to-run
variation. Our port has no equivalent, so leaving the fallback on would
compare two different algorithms rather than two implementations of one.

## One deliberate deviation

Official whisper emits every segment's text with a **leading space**, because
the BPE word-start marker survives into the decoded string. Transcriptor
strips it: the timestamp already supplies the word break, and a stray space is
ugly in `.txt`/`.srt` exports and in concatenated transcripts.

The goldens are stored *already stripped*, so the Rust comparison stays a
plain byte-for-byte check with no normalisation hidden inside the test. The
stripping itself is pinned by its own unit test
(`segment_text_drops_the_post_timestamp_space`), so it cannot silently rot.
Run the generator with `--keep-leading-space` to see the raw reference form.

## Why the test gate is layered

Exact equality is achievable, and we hold it where it matters most: on
**large-v3 transcribing Hebrew** — the model and language people actually use
— the output matches the reference character for character.

The other models are close but not identical, and the pattern is explicable
rather than alarming: our arithmetic is not PyTorch's (burn on f32 against
PyTorch on f32), so where two tokens are nearly tied, a difference in the last
decimal place flips the winner. A 1.55 B-parameter model is confident and
agrees exactly; `tiny` barely decides and drifts.

So `crates/whisper-burn-cli/tests/golden_models.rs` asserts:

1. **Exact** — large-v3 in Hebrew matches character for character.
2. **Structural** — every model detects the language correctly and lands
   within one segment of the reference.
3. **Similarity** — transcripts in the spoken language stay within 10% edit
   distance. Worst observed case is 9 characters on 157 (`tiny`, 5.7%).
4. **Recorded only** — forcing English onto Hebrew audio is deliberately
   excluded. The task is degenerate by design and both implementations emit
   different nonsense; a check there would measure luck, not quality.

Measured at the time of writing, across all twelve configurations:

| configuration | language | segments | worst timestamp delta | text edit distance |
|---|---|---|---|---|
| large-v3 / auto | correct | 3 = 3 | 0 ms | 0 (exact) |
| large-v3 / he | correct | 3 = 3 | 0 ms | 0 (exact) |
| large-v3-turbo / auto | correct | 2 = 2 | 20 ms | 1 |
| small / auto | correct | 1 = 1 | 0 ms | 2 |
| medium / auto | correct | 2 = 2 | 0 ms | 4 |
| base / auto | correct | 1 = 1 | 200 ms | 0 |
| large / auto | correct | 3 = 3 | 1060 ms | 5 |
| tiny / auto, he | correct | 2 = 2 | 5760 ms | 9 |
| large-v2 / auto | correct | **2 vs 3** | — | 2 |
| tiny / en *(recorded)* | correct | 4 = 4 | 0 ms | 63 |
| large-v3 / en *(recorded)* | correct | 2 = 2 | 0 ms | 26 |

Two known outliers, worth revisiting rather than explaining away:

- **large-v2 emits 2 segments where the reference emits 3.**
- **`tiny`'s first timestamp lands 5.7 s away from the reference's.**

## Running the tests

Everything here needs model weights, so the heavy tiers are opt-in:

```text
cargo test -p whisper-burn-cli --test golden_models -- --ignored
```

The first run downloads weights (roughly 14 GB for all eight models) and takes
about 12 minutes. `every_golden_is_present_and_well_formed` and
`edit_distance_is_levenshtein` need no weights and run by default.
