import { illustrationConfig } from './data.js';

const moduleGrid = document.getElementById('module-grid');
const dataIconContainer = document.querySelector('.data-icons');
const connectorLayer = document.querySelector('.connector-layer');
const diagramStage = document.querySelector('.diagram-stage');

const iconTemplates = {
  image: `<svg viewBox="0 0 36 36" aria-hidden="true" focusable="false"><rect x="4" y="6" width="28" height="24" rx="6" ry="6" fill="none" stroke="currentColor" stroke-width="2" /><path d="M10 22l6-6 5 5 5-4 4 5" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" /><circle cx="14" cy="14" r="2.8" fill="currentColor" /></svg>`,
  doc: `<svg viewBox="0 0 36 36" aria-hidden="true" focusable="false"><path d="M9 4h12l6 6v20a2 2 0 0 1-2 2H9a2 2 0 0 1-2-2V6a2 2 0 0 1 2-2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" /><path d="M21 4v8h8" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" /><path d="M12 18h12M12 24h12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>`,
  audio: `<svg viewBox="0 0 36 36" aria-hidden="true" focusable="false"><path d="M12 12l10-6v24l-10-6H8a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2z" fill="none" stroke="currentColor" stroke-width="2" stroke-linejoin="round" /><path d="M26 14a6 6 0 0 1 0 8m-4-10a9 9 0 0 1 0 12" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" /></svg>`,
};

const colorTokens = {
  accent: 'var(--accent)',
  ink: 'var(--ink)',
  'ink-muted': 'var(--ink-muted)',
};

const dashStyles = {
  dashed: '10 8',
  'dashed-arrow': '10 8',
};

const createModuleChip = (module) => {
  const article = document.createElement('article');
  article.className = `module-chip${module.priority === 'high' ? ' module-chip--highlight' : ''}`;
  article.innerHTML = `
    <h3>${module.name}</h3>
    <ul>
      ${module.bullets.map((item) => `<li>${item}</li>`).join('')}
    </ul>
  `;
  return article;
};

const populateModules = () => {
  const fragment = document.createDocumentFragment();
  illustrationConfig.platform.modules.forEach((module) => {
    fragment.appendChild(createModuleChip(module));
  });
  moduleGrid.appendChild(fragment);
};

const populateDataIcons = () => {
  const fragment = document.createDocumentFragment();
  illustrationConfig.leftPane.icons.forEach((key) => {
    const span = document.createElement('span');
    span.className = `data-icon data-icon--${key}`;
    span.innerHTML = iconTemplates[key];
    fragment.appendChild(span);
  });
  dataIconContainer.appendChild(fragment);
};

const anchorVectors = {
  right: { x: 1, y: 0 },
  left: { x: -1, y: 0 },
  top: { x: 0, y: -1 },
  bottom: { x: 0, y: 1 },
};

const getAnchorPoint = (rect, anchor, stageRect) => {
  switch (anchor) {
    case 'right':
      return { x: rect.right - stageRect.left, y: rect.top + rect.height / 2 - stageRect.top };
    case 'left':
      return { x: rect.left - stageRect.left, y: rect.top + rect.height / 2 - stageRect.top };
    case 'top':
      return { x: rect.left + rect.width / 2 - stageRect.left, y: rect.top - stageRect.top };
    case 'bottom':
      return { x: rect.left + rect.width / 2 - stageRect.left, y: rect.bottom - stageRect.top };
    default:
      return { x: rect.left + rect.width / 2 - stageRect.left, y: rect.top + rect.height / 2 - stageRect.top };
  }
};

const ensureDefs = () => {
  let defs = connectorLayer.querySelector('defs');
  if (!defs) {
    defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
    connectorLayer.appendChild(defs);
  }
  let arrowMarker = connectorLayer.querySelector('#arrow-accent');
  if (!arrowMarker) {
    arrowMarker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
    arrowMarker.setAttribute('id', 'arrow-accent');
    arrowMarker.setAttribute('markerWidth', '12');
    arrowMarker.setAttribute('markerHeight', '12');
    arrowMarker.setAttribute('refX', '6');
    arrowMarker.setAttribute('refY', '6');
    arrowMarker.setAttribute('orient', 'auto');
    arrowMarker.setAttribute('markerUnits', 'strokeWidth');

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', 'M2 1l8 5-8 5z');
    path.setAttribute('fill', 'var(--accent)');
    arrowMarker.appendChild(path);
    defs.appendChild(arrowMarker);
  }
};

const drawConnectors = () => {
  ensureDefs();
  const stageRect = diagramStage.getBoundingClientRect();
  connectorLayer.setAttribute('width', stageRect.width);
  connectorLayer.setAttribute('height', stageRect.height);
  connectorLayer.setAttribute('viewBox', `0 0 ${stageRect.width} ${stageRect.height}`);

  while (connectorLayer.lastChild && connectorLayer.lastChild.tagName !== 'defs') {
    connectorLayer.removeChild(connectorLayer.lastChild);
  }

  illustrationConfig.connectors.forEach((connector) => {
    const fromEl = document.querySelector(`[data-node="${connector.from}"]`);
    const toEl = document.querySelector(`[data-node="${connector.to}"]`);
    if (!fromEl || !toEl) return;

    const fromRect = fromEl.getBoundingClientRect();
    const toRect = toEl.getBoundingClientRect();

    const start = getAnchorPoint(fromRect, connector.startAnchor, stageRect);
    const end = getAnchorPoint(toRect, connector.endAnchor, stageRect);

    const vectorStart = anchorVectors[connector.startAnchor] || { x: 0, y: 0 };
    const vectorEnd = anchorVectors[connector.endAnchor] || { x: 0, y: 0 };
    const offset = connector.curvature ?? 80;

    const c1 = {
      x: start.x + vectorStart.x * offset,
      y: start.y + vectorStart.y * offset,
    };
    const c2 = {
      x: end.x - vectorEnd.x * offset,
      y: end.y - vectorEnd.y * offset,
    };

    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('d', `M${start.x} ${start.y} C ${c1.x} ${c1.y}, ${c2.x} ${c2.y}, ${end.x} ${end.y}`);
    path.setAttribute('fill', 'none');
    path.setAttribute('stroke', colorTokens[connector.color] || colorTokens.ink);
    path.setAttribute('stroke-width', connector.width || 2);
    path.setAttribute('stroke-linecap', 'round');
    path.classList.add('connector-path');

    if (connector.style && dashStyles[connector.style]) {
      path.setAttribute('stroke-dasharray', dashStyles[connector.style]);
      path.classList.add('connector-path--dashed');
    }

    if (connector.style === 'dashed-arrow') {
      path.setAttribute('marker-end', 'url(#arrow-accent)');
    }

    connectorLayer.appendChild(path);
  });
};

populateModules();
populateDataIcons();
drawConnectors();

const resizeObserver = new ResizeObserver(() => {
  window.requestAnimationFrame(drawConnectors);
});

resizeObserver.observe(diagramStage);
window.addEventListener('resize', () => window.requestAnimationFrame(drawConnectors));
