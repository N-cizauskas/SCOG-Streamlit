# SCOG GitHub Front Page

This folder hosts the professional static front page for SCOG on GitHub Pages.

It is designed to:

- present the project overview and methodology
- mirror the key content from the Streamlit Home page
- link users into the full Streamlit workflow

## Files

- `index.html` - front page content and structure
- `styles.css` - front page styling
- `app.js` - legacy prototype script (not required by current front page)

## Streamlit link

The front page launch button currently points to:

- `http://localhost:8501`

For deployment, replace it in `index.html` with your hosted Streamlit URL (for example `https://your-app.streamlit.app`).

## Local preview

Open `index.html` directly in a browser.

## GitHub Pages deployment

1. Push `github_site/` to your repository.
2. In repository settings, enable **Pages**.
3. Choose the branch/folder that serves this directory.
4. Use your Pages URL as the project front page and keep Streamlit as the application backend.
