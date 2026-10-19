# Latency

Measured on the development machine, an x86 laptop core, with the int8 model
and the beam search in `runtime/python/transliterator.py`. The numbers will
differ on a phone, but the ratios hold.

## Where the time goes

The decoder takes the **whole target prefix** and returns the logits for the
last position. The prefix is recomputed at every step, so the decoder cost
grows with the square of the output length.

| Setting | ms per word | Top-1 |
|---|---|---|
| Greedy, int8 | **10.5** | 46.3% |
| Beam 4, int8 | 127 | 46.7% |
| Beam 4, fp32 | 193 | 46.9% |

Two things fall out of this.

**Greedy is fast enough for a keyboard.** Ten milliseconds is under a frame
budget of a hundred times over, and the accuracy cost of greedy against beam
search is under half a point. The live path should be greedy.

**Beam search is a deliberate action, not the default.** At 127 milliseconds
it is fine when the user asks for more options and wasteful when they press
space. The keyboard should run greedy, and only run the beam when the user
taps something like "more".

**The prefix recomputation is the thing to fix first.** A key value cache
would turn the decoder from quadratic into linear in the output length, which
should bring beam search down to roughly the greedy cost. It complicates the
exported graph, so it is a later task.

## Quantization

| Format | Size | Top-1 | CER |
|---|---|---|---|
| fp32 | 21.8 MB | 46.87% | 15.65% |
| int8 | **6.1 MB** | 46.70% | 15.65% |

Dynamic weight quantization costs **0.17 points of Top-1 and no measurable
CER**, and the int8 model is 34 percent faster. There is no reason to ship
the fp32 model.

## What this means for the phone

The model is 6.1 MB as int8, which fits comfortably inside an application
package. At greedy speed the inference is negligible next to the cost of
drawing a keyboard.

The remaining unknown is how the numbers translate to a phone. The x86 core
here is not obviously slower than a phone core for this kind of work, but it
has to be measured on the actual device. That is the first thing to do after
the demo is working.
