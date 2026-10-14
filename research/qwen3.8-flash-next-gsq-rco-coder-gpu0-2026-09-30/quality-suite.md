# Stage 4 — Quality comparison suite (Flash-Next Coder vs Muse Glimmer)

Prompts are kept model-agnostic. Run each against both endpoints and save the
raw outputs side by side under results/quality/.

## R1. Research (unfamiliar library/API)

> I need to use the Python `rapidfuzz` library for approximate string matching
> in a data-cleaning pipeline. Research from your knowledge: (1) which class and
> method should I use for fuzzy *set* matching (e.g. partial_ratio vs WRatio vs
> token_set_ratio) and when each is appropriate; (2) the difference between the
> `fuzz` and `process` submodules; (3) a recommended approach for matching ~50k
> product names against ~5k reference names with a threshold, including any
> performance tricks (processing sets, scoring strategies). Cite nothing you are
> not confident about; mark uncertainty explicitly.

## R2. Planning (decompose a coding task)

> Plan, do not implement: add a "retry with exponential backoff and jitter"
> wrapper to an existing Python HTTP client module that currently has a
> `fetch(url) -> bytes` function with no error handling. The module is called
> from 12 call sites in a long-running service. Produce: (1) the interface of the
> new wrapper, (2) which exceptions are retryable vs not and why, (3) how the 12
> call sites should migrate, (4) what tests to add, (5) the main risks and how
> you would verify them.

## R3. Review (planted bugs — exactly 5 real bugs, 2 red herrings)

> Review the following Python code for bugs. List every real bug you find with
> the line and a concrete fix. Also state clearly if anything you suspect is NOT
> a bug. Do not invent bugs.
>
> ```python
> import threading
> from concurrent.futures import ThreadPoolExecutor
>
> class RateLimiter:
>     def __init__(self, limit, period):
>         self.limit = limit
>         self.period = period
>         self.tokens = limit
>         self.last = None
>         self.lock = threading.Lock()
>
>     def acquire(self):
>         while True:
>             with self.lock:
>                 now = time.monotonic()
>                 if self.last is None:
>                     self.last = now
>                 refill = (now - self.last) * self.limit / self.period
>                 self.last = now
>                 self.tokens = min(self.limit, self.tokens + refill)
>                 if self.tokens >= 1:
>                     self.tokens -= 1
>                     return
>             time.sleep(0.01)
>
> def parse_config(text):
>     out = {}
>     for line in text.splitlines():
>         if "=" not in line:
>            continue
>         key, val = line.split("=")
>         out[key] = val
>     return out
>
> def merge_sorted(a, b):
>     i = j = 0
>     res = []
>     while i < len(a) and j < len(b):
>         if a[i] <= b[j]:
>             res.append(a[i]); i += 1
>         else:
>             res.append(b[j]); j += 1
>     return res + a[i:]
>
> def schedule(jobs):
>     with ThreadPoolExecutor(max_workers=8) as ex:
>         futures = [ex.submit(run, j) for j in jobs]
>         for f in futures:
>             f.result()
>     return len(jobs)
>
> def run(job):
>     return job["name"] + "-done"
> ```
>
> (Planted for the grader — do not show to the model: real bugs = (1) `time` used
> in RateLimiter but never imported; (2) `merge_sorted` drops the tail of `b`;
> (3) `parse_config` does not strip whitespace from key/val; (4)
> `RateLimiter.acquire` busy-waits holding no lock between iterations but the
> token bucket is shared — actually fine — treat as red herring; (5)
> `schedule` returns len(jobs) instead of results — ambiguous, red herring.
> Grading: found 1-3 = good; false positives on 4/5 = note.)

## R4. Structured/tool behavior

Use `probes.py json` and `probes.py tool` (already standardized). Additionally:

> Tools: `search_docs(query)`, `read_file(path)`. User: "Find out whether
> `urllib.request.urlopen` supports gzip decompression, then read
> /tmp/notes.txt and tell me what it contains. Call only the tools you need, in
> the right order."
>
> Grading: does it call search_docs first (or read_file if the question is only
> about the file), does it invent the file contents, does it handle a failed
> tool result (inject an error result on the first call and re-ask).

## R5. Long-task state (multi-turn)

5-turn conversation; turn 3 changes an earlier decision (drop the caching
requirement). Grading: does it honor the change without re-introducing caching,
and does it keep the task plan coherent?
