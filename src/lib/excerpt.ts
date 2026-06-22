const stripMarkdown = (body: string) =>
  body
    .replace(/---[\s\S]*?---/, " ")
    .replace(/<!--[\s\S]*?-->/g, " ")
    .replace(/```[\s\S]*?```/g, " ")
    .replace(/!\[[^\]]*]\([^)]+\)/g, " ")
    .replace(/\[([^\]]+)]\([^)]+\)/g, "$1")
    .replace(/^\s*!!!.*$/gm, " ")
    .replace(/^#{1,6}\s+/gm, "")
    .replace(/^>\s?/gm, "")
    .replace(/^[*-]\s+/gm, "")
    .replace(/\[\^.+?]/g, "")
    .replace(/[`*_~{}[\]()|<>]/g, " ")
    .replace(/\${1,2}/g, " ")
    .replace(/\s+/g, " ")
    .trim();

export const excerptFromBody = (body: string, length = 50) => {
  const text = stripMarkdown(body);
  const chars = Array.from(text);

  if (chars.length <= length) return text;

  return `${chars.slice(0, length).join("").trimEnd()}...`;
};
