const calloutPattern = /^(?<marker>!!!|\?\?\?)\s+(?<kind>[A-Za-z0-9_-]+)?(?:\s+(?<title>.+))?$/;
const tabPattern = /^===\s+(?<title>.+)$/;

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;");
}

function nodeText(node) {
  if (!node) return "";
  if (node.type === "text" || node.type === "inlineCode") return node.value;
  if (Array.isArray(node.children)) return node.children.map(nodeText).join("");
  return "";
}

function cleanTitle(value) {
  const trimmed = String(value ?? "").trim();
  const quoted = trimmed.match(/^["'](?<title>.*)["']$/);
  return quoted?.groups?.title ?? trimmed;
}

function markerFromNode(node) {
  if (node.type === "paragraph") {
    const text = nodeText(node).trim();
    return parseMarker(text);
  }

  if (node.type === "code") {
    const text = String(node.value ?? "").trim();
    if (!text.includes("\n")) return parseMarker(text);
  }

  return null;
}

function stripOuterBlankLines(lines) {
  const next = [...lines];
  while (next.length && next[0].trim() === "") next.shift();
  while (next.length && next.at(-1).trim() === "") next.pop();
  return next;
}

function stripCommonIndent(lines) {
  const indents = lines
    .filter((line) => line.trim() !== "")
    .map((line) => line.match(/^\s*/)?.[0].length ?? 0);
  const indent = indents.length ? Math.min(...indents) : 0;
  return indent > 0 ? lines.map((line) => line.slice(indent)) : lines;
}

function nodeFromCodeLines(lines, fallbackNode) {
  const normalized = stripCommonIndent(stripOuterBlankLines(lines));
  if (!normalized.length) return null;

  const first = normalized[0].match(/^```(?<lang>[A-Za-z0-9_-]+)?(?:\s+.*)?$/);
  const last = normalized.at(-1)?.match(/^```\s*$/);

  if (first && last) {
    return {
      type: "code",
      lang: first.groups?.lang || null,
      value: normalized.slice(1, -1).join("\n")
    };
  }

  return { ...fallbackNode, value: normalized.join("\n") };
}

function expandCodeNode(node) {
  if (node.type !== "code") return [node];

  const lines = String(node.value ?? "").split("\n");
  const hasMkdocsMarker = lines.some((line) => parseMarker(line.trim()));
  const fencedOnly = nodeFromCodeLines(lines, node);

  if (!hasMkdocsMarker) return fencedOnly ? [fencedOnly] : [node];

  const expanded = [];
  let buffer = [];

  const flushBuffer = () => {
    const next = nodeFromCodeLines(buffer, node);
    if (next) expanded.push(next);
    buffer = [];
  };

  for (const line of lines) {
    const marker = parseMarker(line.trim());
    if (marker) {
      flushBuffer();
      expanded.push({ type: "paragraph", children: [{ type: "text", value: line.trim() }] });
      continue;
    }

    buffer.push(line);
  }

  flushBuffer();
  return expanded;
}

function parseMarker(text) {
  const callout = text.match(calloutPattern);
  if (callout?.groups) {
    const marker = callout.groups.marker;
    const kind = (callout.groups.kind || "note").toLowerCase();
    const title = cleanTitle(callout.groups.title || callout.groups.kind || "Note");
    return { type: "callout", collapsible: marker === "???", kind, title };
  }

  const tab = text.match(tabPattern);
  if (tab?.groups) {
    return { type: "tab", title: cleanTitle(tab.groups.title) };
  }

  return null;
}

function openCallout(marker) {
  const kind = escapeHtml(marker.kind);
  const title = escapeHtml(marker.title);

  if (marker.collapsible) {
    return {
      type: "html",
      value: `<details class="mk-callout mk-callout-${kind}"><summary class="mk-callout-title">${title}</summary><div class="mk-callout-body">`
    };
  }

  return {
    type: "html",
    value: `<section class="mk-callout mk-callout-${kind}"><div class="mk-callout-title">${title}</div><div class="mk-callout-body">`
  };
}

function closeCallout(collapsible) {
  return { type: "html", value: collapsible ? "</div></details>" : "</div></section>" };
}

function openTab(marker) {
  return {
    type: "html",
    value: `<section class="mk-tab-card"><div class="mk-tab-title">${escapeHtml(marker.title)}</div><div class="mk-tab-body">`
  };
}

function closeTab() {
  return { type: "html", value: "</div></section>" };
}

function transformChildren(children) {
  const next = [];
  let activeCallout = null;
  let activeTab = false;

  const closeActiveTab = () => {
    if (!activeTab) return;
    next.push(closeTab());
    activeTab = false;
  };

  const closeActiveCallout = () => {
    if (!activeCallout) return;
    next.push(closeCallout(activeCallout.collapsible));
    activeCallout = null;
  };

  for (const child of children) {
    const expandedChildren = expandCodeNode(child);

    for (const expandedChild of expandedChildren) {
    const marker = markerFromNode(expandedChild);

    if (marker?.type === "callout") {
      closeActiveTab();
      closeActiveCallout();
      next.push(openCallout(marker));
      activeCallout = marker;
      continue;
    }

    if (marker?.type === "tab") {
      closeActiveTab();
      next.push(openTab(marker));
      activeTab = true;
      continue;
    }

    if (expandedChild.type === "heading") {
      closeActiveTab();
      closeActiveCallout();
    }

    next.push(expandedChild);
    }
  }

  closeActiveTab();
  closeActiveCallout();

  return next;
}

export default function remarkMkdocsBlocks() {
  return (tree) => {
    if (!Array.isArray(tree.children)) return;
    tree.children = transformChildren(tree.children);
  };
}
