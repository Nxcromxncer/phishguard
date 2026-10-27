"""Generate a SYNTHETIC labelled URL file for exercising the pipeline offline.

*** THIS IS NOT REAL DATA. ***  The URLs are invented from templates, are easy to
tell apart by construction, and say nothing about real phishing. Metrics from a
model trained on it are meaningless as a measure of detection quality. It exists
so the prepare -> train -> evaluate -> API flow can be tested without network
access or a licensed dataset. Rows carry ``source=synthetic``, which makes the
dataset report, the model metadata, and the API all label the model as a demo
(and the backend refuses to load it unless explicitly allowed).

No URL generated here points at a real phishing site; hosts use invented names.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

BRANDS = ["paypal", "apple", "microsoft", "google", "amazon", "netflix", "dropbox", "linkedin"]
KEYWORDS = ["login", "verify", "secure", "account", "update", "confirm", "billing", "signin"]
WORDS = ["garden", "river", "stone", "bright", "cloud", "maple", "north", "harbor", "pixel", "summit",
         "lumen", "orchard", "falcon", "meadow", "copper", "anchor", "violet", "ember", "quartz", "willow"]
TLDS_LEGIT = ["com", "org", "net", "io", "co.uk", "edu"]
TLDS_PHISH = ["com", "net", "xyz", "top", "info", "online", "site", "co.uk"]
LEGIT_PATHS = ["", "about", "products/list", "blog/2024/news", "docs/guide", "support/contact", "login", "account", "help/faq"]


def rand_token(rng: random.Random, n: int = 8) -> str:
    return "".join(rng.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(n))


def phishing_url(rng: random.Random) -> str:
    kind = rng.choice(["brand_host", "ip", "userinfo", "deep", "plain", "keyword_path"])
    scheme = rng.choices(["http", "https"], weights=[2, 3])[0]
    if kind == "brand_host":
        host = f"{rng.choice(BRANDS)}-{rng.choice(KEYWORDS)}-{rand_token(rng, 5)}.{rng.choice(TLDS_PHISH)}"
        return f"{scheme}://{host}/{rng.choice(KEYWORDS)}/{rand_token(rng)}.php?id={rng.randint(1000, 99999)}"
    if kind == "ip":
        ip = ".".join(str(rng.randint(11, 223)) for _ in range(4))
        return f"http://{ip}/{rng.choice(KEYWORDS)}/index.html"
    if kind == "userinfo":
        return f"{scheme}://{rng.choice(BRANDS)}.com@{rand_token(rng, 9)}.{rng.choice(TLDS_PHISH)}/{rng.choice(KEYWORDS)}"
    if kind == "deep":
        return (f"{scheme}://{rng.choice(BRANDS)}.{rng.choice(KEYWORDS)}.{rand_token(rng, 4)}."
                f"{rand_token(rng, 7)}.{rng.choice(TLDS_PHISH)}/{rand_token(rng, 12)}")
    if kind == "keyword_path":
        return f"{scheme}://{rand_token(rng, 10)}.{rng.choice(TLDS_PHISH)}/{rng.choice(KEYWORDS)}-{rng.choice(KEYWORDS)}/{rand_token(rng, 6)}"
    return f"{scheme}://{rand_token(rng, 11)}.{rng.choice(TLDS_PHISH)}/{rand_token(rng, 6)}"


def legit_url(rng: random.Random) -> str:
    host = f"{rng.choice(WORDS)}{rng.choice(WORDS)}.{rng.choice(TLDS_LEGIT)}"
    if rng.random() < 0.6:
        host = "www." + host
    elif rng.random() < 0.2:
        host = f"{rng.choice(['blog', 'shop', 'docs', 'support'])}." + host
    if rng.random() < 0.15:
        host = host.replace(".", "-", 1) if host.count(".") > 1 else host
    path = rng.choice(LEGIT_PATHS)
    query = f"?ref={rng.choice(WORDS)}" if rng.random() < 0.15 else ""
    return f"https://{host}/{path}{query}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Write a SYNTHETIC url,label,source CSV (pipeline testing only).")
    parser.add_argument("--n-per-class", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rng = random.Random(args.seed)
    rows = [(phishing_url(rng), "phishing") for _ in range(args.n_per_class)]
    rows += [(legit_url(rng), "legitimate") for _ in range(args.n_per_class)]
    rng.shuffle(rows)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["url", "label", "source"])
        writer.writerows((url, label, "synthetic") for url, label in rows)
    print(f"Wrote {len(rows)} SYNTHETIC rows to {args.out} (pipeline testing only; not real data).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
