const escapeHtml = (value) =>
  value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;");

const transformChildren = (node) => {
  if (!Array.isArray(node.children)) return;

  node.children = node.children.map((child) => {
    if (child.type === "code" && child.lang === "mermaid") {
      return {
        type: "html",
        value: `<div class="mermaid" data-mermaid-source="markdown">${escapeHtml(
          child.value.trim()
        )}</div>`
      };
    }

    transformChildren(child);
    return child;
  });
};

export default function remarkMermaidBlocks() {
  return transformChildren;
}
