---
name: Publish HTML to GitHub Pages
overview: Extract the embedded HTML page from `construction_server_fixed.py` into a standalone `index.html` at the repo root and configure the GitHub Actions workflow to deploy it as a static site via GitHub Pages.
todos:
  - id: extract-html
    content: Extract HTML from construction_server_fixed.py into docs/index.html, converting Python f-string escaping and replacing server-side timestamp with client-side JS
    status: completed
  - id: revert-readme
    content: Remove Jekyll front matter from README.md
    status: completed
  - id: update-workflow
    content: Replace Jekyll build in jekyll-gh-pages.yml with static HTML upload from docs/
    status: completed
  - id: delete-config
    content: Delete _config.yml
    status: completed
  - id: cleanup-gitignore
    content: Remove Jekyll-specific entries from .gitignore
    status: completed
isProject: false
---

# Publish HTML to GitHub Pages

## Problem

The current GitHub Pages setup tries to render `README.md` via Jekyll. The user actually wants the HTML web app (stock lookup, leaderboard, analysis UI) that lives inside `construction_server_fixed.py` (lines 1150-1776) to be the published page.

## Changes

### 1. Create `index.html` at the repo root

Extract the HTML from `serve_construction_page()` in [construction_server_fixed.py](construction_server_fixed.py) (lines 1150-1776) into a standalone `index.html`.

Key adjustments needed during extraction:

- Remove Python f-string escaping (`{{` / `}}` become `{` / `}`)
- Replace the server-side timestamp `{datetime.datetime.now().strftime(...)}` with client-side JS that renders the current time
- The API fetch calls (`/lookup`, `/analyze`, `/api/v1/model-leaderboard`) stay as relative URLs -- they won't resolve on GitHub Pages, but the static shell will render correctly

### 2. Revert `README.md`

Remove the Jekyll front matter (`layout`, `title`, `permalink`) that was added previously. Return it to a plain markdown file so it only serves as the repo README, not as the site index.

### 3. Update `_config.yml`

Remove the `remote_theme` and `plugins` since we're serving a plain HTML file, not a Jekyll site. Keep the `exclude` list so non-site files don't get bundled into the deployment artifact.

Alternatively, delete `_config.yml` entirely and switch the workflow away from Jekyll.

### 4. Switch the GitHub Actions workflow to static deployment

Replace the Jekyll build step in [.github/workflows/jekyll-gh-pages.yml](.github/workflows/jekyll-gh-pages.yml) with the simpler **static HTML** pattern -- just upload `index.html` (and optionally `README.md`) directly via `actions/upload-pages-artifact`, skipping the Jekyll build entirely. This avoids needing `_config.yml` at all.

Updated workflow outline:

```yaml
steps:
  - uses: actions/checkout@v4
  - uses: actions/configure-pages@v5
  - uses: actions/upload-pages-artifact@v3
    with:
      path: ./docs   # or a dedicated folder
  - uses: actions/deploy-pages@v4
```

We'll put `index.html` in a `docs/` folder (or use the root with a carefully scoped path) so only the web page gets deployed and the Python source is excluded.

### 5. Clean up

- Delete `_config.yml` (no longer needed)
- Remove Jekyll-related entries from `.gitignore` (`_site/`, `.jekyll-cache/`, etc.) since Jekyll is no longer used

