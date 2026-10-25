---
title: Development and website maintenance
description: Build and validate Rula documentation independently of GPU services, preserving canonical URLs, author identity and accurate search metadata.
---
# Development and website maintenance

The public website is built from `website/` using Zensical with the same theme and metadata conventions as [Rokid Physical AI](https://ykshv.github.io/rokid-physical-ai/). Runtime code remains in `apps/`; the documentation environment does not install models or start the voice backend.

## Documentation checks

Use Python 3.14+ for the root documentation project:

```sh
uv sync --locked
uv run python tools/check_docs.py
uv run zensical build --strict
uv run python tools/check_publication.py
```

The Pages workflow checks pull requests and deploys the `main` branch. The publication check validates each navigation page's canonical URL, unique title/description, social metadata, visible author link, author identity in structured data and exact sitemap coverage. Deployment alone does not prove that a search engine has indexed the pages.

## Search and authorship

Use `https://ykshv.github.io/rula/` as the canonical project website and `https://ykshv.github.io/rula/sitemap.xml` as its sitemap. Keep the GitHub About URL consistent. The interactive runtime endpoint is separate and can be unavailable without taking documentation offline.

Keep author links visible and useful. The homepage's `SoftwareSourceCode` data identifies the repository and the same author entity as the personal site, `https://igor-ya.ru/#person`. Structured data must match visible content; do not add invented reviews, results or an article that does not exist.

The project lives under a GitHub Pages path. Crawler policy belongs to the origin's `/robots.txt`, not `/rula/robots.txt`; a file inside the project path cannot override origin policy. A missing origin robots file does not itself block indexing. Keep canonical pages free of `noindex` and accessible without JavaScript or authentication.

After deployment, inspect the public HTML, response headers and sitemap. A maintainer with Search Console or Bing Webmaster Tools access can submit the sitemap and inspect the canonical URLs. Search inclusion and citations depend on the search engine; a successful HTTP check establishes accessibility, not ranking.
