import configText from "@/research.config.yaml?raw";

export function getResearchVisibility() {
  const projects: Record<string, boolean> = {};
  let inProjects = false;
  let inOrder = false;

  for (const rawLine of configText.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    if (line === "projects:") {
      inProjects = true;
      inOrder = false;
      continue;
    }

    if (line === "order:") {
      inProjects = false;
      inOrder = true;
      continue;
    }

    if (!inProjects || inOrder) {
      continue;
    }

    const match = line.match(/^([A-Za-z0-9_-]+):\s*(true|false)$/i);
    if (match) {
      projects[match[1]] = match[2].toLowerCase() === "true";
    }
  }

  return projects;
}

export function getResearchOrder() {
  const order: string[] = [];
  let inOrder = false;

  for (const rawLine of configText.split(/\r?\n/)) {
    const line = rawLine.trim();

    if (!line || line.startsWith("#")) {
      continue;
    }

    if (line === "order:") {
      inOrder = true;
      continue;
    }

    if (line.endsWith(":")) {
      inOrder = false;
      continue;
    }

    if (!inOrder) {
      continue;
    }

    const match = line.match(/^-\s*([A-Za-z0-9_-]+)$/);
    if (match) {
      order.push(match[1]);
    }
  }

  return order;
}

export function isResearchVisible(id: string) {
  const visibility = getResearchVisibility();
  return visibility[id] !== false;
}

type ResearchProject = {
  id: string;
  data: {
    title: string;
    year: string;
  };
};

export function sortResearchProjects<T extends ResearchProject>(projects: T[]) {
  const order = getResearchOrder();
  const rank = new Map(order.map((id, index) => [id, index]));

  return [...projects].sort((a, b) => {
    const aRank = rank.get(a.id);
    const bRank = rank.get(b.id);

    if (aRank !== undefined || bRank !== undefined) {
      return (aRank ?? Number.MAX_SAFE_INTEGER) - (bRank ?? Number.MAX_SAFE_INTEGER);
    }

    const yearDiff = Number(b.data.year) - Number(a.data.year);
    if (yearDiff !== 0) {
      return yearDiff;
    }

    return a.data.title.localeCompare(b.data.title);
  });
}
