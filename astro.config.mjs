import { defineConfig } from "astro/config";
import rehypeKatex from "rehype-katex";
import remarkMkdocsBlocks from "./src/plugins/remark-mkdocs-blocks.mjs";
import remarkMath from "remark-math";
import stripMarkdownImageAttributes from "./src/plugins/remark-strip-markdown-image-attributes.mjs";

export default defineConfig({
  site: "https://simonjisu.github.io",
  markdown: {
    remarkPlugins: [
      [remarkMath, { singleDollarTextMath: true }],
      remarkMkdocsBlocks,
      stripMarkdownImageAttributes
    ],
    rehypePlugins: [[rehypeKatex, { strict: false }]]
  }
});
