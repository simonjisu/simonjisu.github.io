import { getCollection } from "astro:content";
import { excerptFromBody } from "@/lib/excerpt";

const escapeXml = (value: string) =>
  value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&apos;");

const formatTitle = (id: string) =>
  id
    .replace(/^\d{4}-\d{2}-\d{2}-/, "")
    .replace(/[-_]/g, " ")
    .replace(/\b\w/g, (char) => char.toUpperCase());

export async function GET(context: { site?: URL }) {
  const site = context.site ?? new URL("https://simonjisu.github.io");
  const posts = (await getCollection("writing", ({ data }) => !data.draft))
    .filter((post) => post.data.date)
    .sort((a, b) => {
      const aDate = new Date(a.data.date as string | Date).getTime();
      const bDate = new Date(b.data.date as string | Date).getTime();
      return bDate - aDate;
    });

  const items = posts
    .map((post) => {
      const url = new URL(`/writing/${post.id}/`, site).toString();
      const title = post.data.title ?? formatTitle(post.id);
      const description =
        post.data.description ?? excerptFromBody(post.body);
      const pubDate = new Date(post.data.date as string | Date).toUTCString();

      return `
        <item>
          <title>${escapeXml(title)}</title>
          <link>${escapeXml(url)}</link>
          <guid>${escapeXml(url)}</guid>
          <description>${escapeXml(description)}</description>
          <pubDate>${escapeXml(pubDate)}</pubDate>
        </item>`;
    })
    .join("");

  const xml = `<?xml version="1.0" encoding="UTF-8"?>
    <rss version="2.0">
      <channel>
        <title>Soopace Writing</title>
        <link>${escapeXml(site.toString())}</link>
        <description>Writing and notes by Simon Jisoo Jang.</description>
        ${items}
      </channel>
    </rss>`;

  return new Response(xml, {
    headers: {
      "Content-Type": "application/rss+xml; charset=utf-8"
    }
  });
}
