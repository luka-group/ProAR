# ProAR project page

A standalone academic project page. No package installation or build step is required.

From this directory:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Open http://localhost:8000 in your browser. Stop the server with Ctrl+C.

- `index.html`: text, figures, author information, and citation.
- `styles.css`: colors, typography, and responsive layout.
- `script.js`: task ordering, video pairing, pagination, and synchronized playback.
- `images/*.png`: browser-readable previews of the original PDF figures.
- `images/posters/`: static video previews for lazy loading.
- `paper.pdf`: the supplied manuscript.


