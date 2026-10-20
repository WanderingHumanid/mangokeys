# Latency

## Where the time went

The first version of the browser demo ran the model on the main thread with a
beam search that never stopped early. Two things were wrong.

**The search kept going after the answer was known.** The decoder produces a
candidate at every step, and the search only stopped when every beam had
emitted the end token. A word like `kerala` has a four character answer, but
the weaker beams kept extending, so the search ran for many more steps than
were useful.

The fix is exact pruning. A live beam can only add tokens, and every token has
a log probability of at most zero, so **a live beam's score can never go up.**
That makes it safe to drop a live beam whose score is already behind the best
finished hypothesis, and to stop the whole search when no live beam can still
win. Nothing is approximated, the top result is the same.

**The model ran on the main thread.** Even a fast inference blocks the page,
because the browser cannot paint while javascript is running. The model now
runs in a Web Worker, the rule engine stays on the main thread for an instant
preview, and only the newest request is kept when several arrive.

## The numbers

Measured in headless Chromium on the development machine, with the int8 model.

| | Before | After |
|---|---|---|
| Model, median per word | 166 ms | **60 ms** |
| Model, mean per word | 270 ms | **73 ms** |
| Longest word, `kerala` | 724 ms | **39 ms** |
| Longest frame gap while typing | the whole inference | **17 to 27 ms** |
| Round trip from keystroke to output | 1 s or more | **118 to 246 ms** |

The frame gap is the number that matters for how it feels. A gap of 17 to 27
milliseconds is the normal frame budget of a 60 hertz display, so the page
never stutters. The harness was checked against a deliberate 300 millisecond
block, which it reported as a 316 millisecond gap.

The round trip includes the 80 millisecond typing debounce.

## Threads

The wasm runtime was running on a single core because the browser will not
hand out `SharedArrayBuffer` without cross origin isolation. The demo server
now sends the two headers that enable it:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

With those in place the worker uses up to four threads.

## The same fix applies everywhere

The pruning is in the search, not in the page, so all three runtimes got it:
the Python one, the browser one and the Kotlin one. The Python path over the
whole Aksharantar test set went from 24.5 milliseconds per word to 3.95, and
top-1 accuracy went slightly up.

| | Top-1 | CER | ms / word |
|---|---|---|---|
| Before pruning | 53.30% | 17.06% | 24.49 |
| After pruning | **53.44%** | **16.97%** | **3.95** |

The top-5 number went down, from 72.2% to 64.6%, and that is a correction
rather than a regression. The old figure counted the same text several times
when different beams produced it, so it was measuring duplicates. The top-5 is
now five distinct candidates.

## What is left

**The decoder recomputes the whole prefix at every step**, so the cost grows
with the square of the output length. A key value cache would make it linear.
That is the next thing to do if the latency is not good enough on a phone.

**The model is not trimmed for the browser.** The encoder always runs over 48
source positions even for a five character word. The measurement shows the
encoder is only 4 to 30 milliseconds, so it is not worth the work of making
the sequence length dynamic.

**Beam search is still four times the work of greedy.** The live path could
use greedy and reserve the beam for an explicit request. The cost of greedy
is 0.28 points of top-1, which is not much.
