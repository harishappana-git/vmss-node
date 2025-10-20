# LLM Training Platform Explorer

This project renders a 2D, zoomable visualization that describes the layered architecture of a large-scale LLM distributed training platform. It is a static web experience built with plain HTML, CSS, and JavaScript (D3.js).

## Prerequisites

- A modern web browser (Chrome, Edge, Firefox, or Safari) with JavaScript enabled.
- A simple HTTP server for local development. No additional build tools or package managers are required.

## Getting Started

1. **Clone the repository** (if you have not already):
   ```bash
   git clone <repo-url>
   cd vmss-node
   ```

2. **Start a local static server.** Any basic server works. For example, using Python 3:
   ```bash
   python3 -m http.server 8000
   ```
   This command serves the project files at `http://localhost:8000/`.

   > **Tip:** If you prefer Node.js tooling, you can use `npx serve` or any equivalent static server.

3. **Open the app in your browser:**
   Navigate to `http://localhost:8000/index.html` and interact with the visualization. Hover, zoom, and double-click the different layers to explore their detailed narratives, advances, and challenges.

## Project Structure

```
├── index.html      # Application shell and layout
├── styles.css      # Global styles and theming
└── src
    ├── app.js      # D3-based visualization logic and interactions
    └── data.js     # Layer definitions, narratives, advances, and challenges
```

## Troubleshooting

- **White screen or missing data:** Ensure you are loading the site via `http://` or `https://`. Opening the `index.html` file directly from disk (`file://`) may block module loading in some browsers.
- **Port already in use:** If port 8000 is busy, choose another available port (e.g., `python3 -m http.server 3000`).

## License

This project does not currently include an explicit license. Please contact the repository owner for usage guidelines.
