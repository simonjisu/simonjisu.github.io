const attributePattern = /^\s*\{[^}]*\}\s*$/;

function stripImageAttributeText(node) {
  if (!node || !Array.isArray(node.children)) return;

  node.children = node.children.filter((child, index, children) => {
    const previous = children[index - 1];
    return !(
      previous?.type === "image" &&
      child.type === "text" &&
      attributePattern.test(child.value)
    );
  });

  for (const child of node.children) {
    stripImageAttributeText(child);
  }
}

export default function stripMarkdownImageAttributes() {
  return (tree) => {
    stripImageAttributeText(tree);
  };
}
