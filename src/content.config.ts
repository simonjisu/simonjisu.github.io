import { defineCollection } from "astro:content";
import { glob } from "astro/loaders";
import { z } from "astro/zod";

const linkSchema = z.object({
  label: z.string(),
  href: z.string()
});

const researchFigureSchema = z.object({
  src: z.string(),
  alt: z.string(),
  width: z.string().optional(),
  captionKo: z.string().optional(),
  captionEn: z.string().optional(),
  creditHtml: z.string().optional()
});

const researchTableSchema = z.object({
  captionKo: z.string().optional(),
  captionEn: z.string().optional(),
  columns: z.array(
    z.object({
      key: z.string(),
      labelKo: z.string(),
      labelEn: z.string()
    })
  ),
  rows: z.array(z.record(z.string(), z.string()))
});

const researchSectionSchema = z.object({
  key: z.string(),
  navKo: z.string(),
  navEn: z.string(),
  titleKo: z.string(),
  titleEn: z.string(),
  bodyKo: z.array(z.string()),
  bodyEn: z.array(z.string()),
  highlightsKo: z.array(z.string()).default([]),
  highlightsEn: z.array(z.string()).default([]),
  table: researchTableSchema.optional(),
  figure: researchFigureSchema.optional(),
  gallery: z.array(researchFigureSchema).default([])
});

const demoVideoSchema = z.object({
  title: z.string(),
  titleKo: z.string().optional(),
  titleEn: z.string().optional(),
  url: z.string(),
  embedUrl: z.string(),
  descriptionKo: z.string(),
  descriptionEn: z.string()
});

const referenceSchema = z.object({
  text: z.string(),
  href: z.string().optional()
});

const research = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/research" }),
  schema: z.object({
    title: z.string(),
    subtitle: z.string(),
    subtitleKo: z.string().optional(),
    year: z.string(),
    status: z.string(),
    venue: z.string().optional(),
    image: z.string(),
    showHeroImage: z.boolean().default(true),
    summary: z.string(),
    summaryKo: z.string().optional(),
    sourcePdf: z.string().optional(),
    sourceFolder: z.string().optional(),
    links: z.array(linkSchema).default([]),
    demoVideos: z.array(demoVideoSchema).default([]),
    references: z.array(referenceSchema).default([]),
    sections: z.array(researchSectionSchema).min(4).max(4)
  })
});

const writing = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/writing" }),
  schema: z
    .object({
      title: z.string().optional(),
      description: z.string().optional(),
      date: z.union([z.string(), z.date()]).optional(),
      draft: z.boolean().optional(),
      authors: z.array(z.string()).optional(),
      categories: z.array(z.string()).optional(),
      tags: z.array(z.string()).optional()
    })
    .passthrough()
});

const notes = defineCollection({
  loader: glob({ pattern: "**/*.{md,mdx}", base: "./src/content/notes" }),
  schema: z
    .object({
      title: z.string().optional(),
      description: z.string().optional(),
      date: z.union([z.string(), z.date()]).optional(),
      draft: z.boolean().optional(),
      tags: z.array(z.string()).optional()
    })
    .passthrough()
});

export const collections = { research, writing, notes };
