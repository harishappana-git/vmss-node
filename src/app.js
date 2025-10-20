import { layers } from './data.js';

const svg = d3.select('#architecture-svg');
const bounds = svg.node().getBoundingClientRect();
const width = bounds.width || 960;
const height = bounds.height || 1200;

svg.attr('viewBox', `0 0 ${width} ${height}`).attr('preserveAspectRatio', 'xMidYMid meet');

const margins = { top: 40, right: 60, bottom: 60, left: 60 };
const layerWidth = Math.min(720, width - margins.left - margins.right);
const layerHeight = 80;
const layerSpacing = 60;

const zoomLayer = svg.append('g').attr('class', 'zoom-layer');

const zoomBehavior = d3
  .zoom()
  .scaleExtent([0.7, 5])
  .on('zoom', (event) => {
    zoomLayer.attr('transform', event.transform);
  });

svg.call(zoomBehavior);
svg.on('dblclick.zoom', null); // disable built-in double click zoom to customize

const tooltip = d3
  .select('.visualization')
  .append('div')
  .attr('class', 'tooltip');

const computePositions = () => {
  const startY = margins.top;
  return layers.map((layer, index) => {
    const x = width / 2;
    const y = startY + index * (layerHeight + layerSpacing) + layerHeight / 2;
    return { ...layer, x, y };
  });
};

const positionedLayers = computePositions();

// Draw connectors between layers
zoomLayer
  .selectAll('path.connector')
  .data(d3.pairs(positionedLayers))
  .enter()
  .append('path')
  .attr('class', 'connector')
  .attr('d', ([from, to]) => {
    const controlOffset = 40;
    return `M${from.x} ${from.y + layerHeight / 2}
            C ${from.x} ${from.y + layerHeight / 2 + controlOffset},
              ${to.x} ${to.y - layerHeight / 2 - controlOffset},
              ${to.x} ${to.y - layerHeight / 2}`;
  });

const layerGroups = zoomLayer
  .selectAll('g.layer')
  .data(positionedLayers)
  .enter()
  .append('g')
  .attr('class', 'layer')
  .attr('transform', (d) => `translate(${d.x - layerWidth / 2}, ${d.y - layerHeight / 2})`)
  .on('mouseenter', function (event, layer) {
    d3.select(this).select('.layer-card').classed('hovered', true);
    tooltip
      .classed('visible', true)
      .text(layer.summary)
      .style('left', `${event.offsetX}px`)
      .style('top', `${event.offsetY}px`);
  })
  .on('mousemove', (event) => {
    tooltip.style('left', `${event.offsetX}px`).style('top', `${event.offsetY}px`);
  })
  .on('mouseleave', function () {
    d3.select(this).select('.layer-card').classed('hovered', false);
    tooltip.classed('visible', false);
  })
  .on('click', (event) => {
    event.stopPropagation();
  })
  .on('dblclick', (event, layer) => {
    event.stopPropagation();
    focusLayer(layer);
  });

layerGroups
  .append('rect')
  .attr('class', 'layer-card')
  .attr('width', layerWidth)
  .attr('height', layerHeight);

layerGroups
  .append('text')
  .attr('class', 'layer-label')
  .attr('x', layerWidth / 2)
  .attr('y', 26)
  .attr('text-anchor', 'middle')
  .text((d) => d.name);

// Add mini components inside each layer for extra fidelity
const subComponentHeight = 22;
const subComponentPadding = 16;

layerGroups.each(function (layer) {
  const group = d3.select(this);
  const items = layer.advances.slice(0, 2); // show highlights inside card
  const cardWidth = layerWidth - subComponentPadding * 2;

  items.forEach((text, index) => {
    const subY = 38 + index * (subComponentHeight + 8);
    const subGroup = group
      .append('g')
      .attr('class', 'sub-component')
      .attr('transform', `translate(${subComponentPadding}, ${subY})`);

    subGroup
      .append('rect')
      .attr('width', cardWidth)
      .attr('height', subComponentHeight)
      .attr('rx', 8)
      .attr('ry', 8);

    subGroup
      .append('text')
      .attr('x', 10)
      .attr('y', subComponentHeight / 2 + 1)
      .attr('dominant-baseline', 'middle')
      .text(text);
  });
});

// Panel references
const detailsTitle = document.getElementById('details-title');
const detailsDescription = document.getElementById('details-description');
const detailsAdvances = document.getElementById('details-advances');
const detailsChallenges = document.getElementById('details-challenges');
const detailsNarrative = document.getElementById('details-narrative');

function populateDetails(layer) {
  detailsTitle.textContent = layer.name;
  detailsDescription.textContent = layer.summary;

  renderList(detailsAdvances, 'Key Advances & Timeline', layer.advances);
  renderList(detailsChallenges, 'Open Challenges at Hyperscale', layer.challenges);
  renderNarrative(detailsNarrative, layer.deepDive);
}

let activeLayerId = null;

function renderList(container, title, items = []) {
  container.replaceChildren();
  if (!items.length) {
    return;
  }

  const heading = document.createElement('h3');
  heading.textContent = title;
  const list = document.createElement('ul');
  items.forEach((item) => {
    const li = document.createElement('li');
    li.textContent = item;
    list.appendChild(li);
  });

  container.appendChild(heading);
  container.appendChild(list);
}

function renderNarrative(container, sections = []) {
  container.replaceChildren();
  if (!sections.length) {
    return;
  }

  const fragment = document.createDocumentFragment();

  sections.forEach((section) => {
    const block = document.createElement('div');
    block.className = 'narrative-section';

    if (section.heading) {
      const heading = document.createElement('h3');
      heading.textContent = section.heading;
      block.appendChild(heading);
    }

    (section.paragraphs || []).forEach((paragraph) => {
      const p = document.createElement('p');
      p.textContent = paragraph;
      block.appendChild(p);
    });

    fragment.appendChild(block);
  });

  container.appendChild(fragment);
}

function focusLayer(layer) {
  const scale = 2.2;
  const centerX = width / 2;
  const centerY = height / 2;
  const translateX = centerX - layer.x * scale;
  const translateY = centerY - layer.y * scale;

  svg
    .transition()
    .duration(750)
    .call(zoomBehavior.transform, d3.zoomIdentity.translate(translateX, translateY).scale(scale));

  layerGroups.selectAll('.layer-card').classed('active', (d) => d.id === layer.id);
  populateDetails(layer);
  activeLayerId = layer.id;
}

function resetFocus() {
  svg.transition().duration(600).call(zoomBehavior.transform, d3.zoomIdentity);
  layerGroups.selectAll('.layer-card').classed('active', false);
  activeLayerId = null;
  detailsTitle.textContent = 'Layer overview';
  detailsDescription.textContent =
    'Select a layer to examine its role in orchestrating large-scale model training.';
  detailsAdvances.replaceChildren();
  detailsChallenges.replaceChildren();
  detailsNarrative.replaceChildren();
}

svg.on('dblclick', () => {
  if (activeLayerId !== null) {
    resetFocus();
  }
});

// Initialize with top layer for context
focusLayer(positionedLayers[0]);
