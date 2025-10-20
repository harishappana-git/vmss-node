# LLM Training Platform Illustration

This project renders a single-page, isometric technical illustration that captures the layered architecture of a large-scale LLM distributed training platform. The experience is a static web page built with HTML, CSS, and vanilla JavaScript and does not require any build tooling.

## Prerequisites

- A modern web browser (Chrome, Edge, Firefox, or Safari) with JavaScript enabled.
- A simple HTTP server for local development. No additional package manager is required.

## Getting Started

1. **Clone the repository** (if you have not already):
   ```bash
   git clone <repo-url>
   cd vmss-node
   ```

2. **Start a local static server.** Any lightweight server works. For example, using Python 3:
   ```bash
   python3 -m http.server 8000
   ```
   This command serves the project files at `http://localhost:8000/`.

   > **Tip:** If you prefer Node.js tooling, you can use `npx serve` or any similar static server.

3. **Open the illustration in your browser:**
   Navigate to `http://localhost:8000/index.html` to view the isometric diagram that summarizes each layer of the training stack.

## Project Structure

```
├── index.html      # Application shell and semantic layout
├── styles.css      # Global design tokens and isometric styling
└── src
    ├── app.js      # DOM assembly, icon rendering, connector layout
    └── data.js     # Configuration for modules and connector routing
```

## Troubleshooting

- **Blank page or missing connectors:** Ensure you are loading the page via `http://` or `https://`. Browsers may block ES module loading for files opened directly via `file://`.
- **Port already in use:** If port 8000 is busy, choose another available port (e.g., `python3 -m http.server 3000`).

## License

This project does not currently include an explicit license. Please contact the repository owner for usage guidelines.
