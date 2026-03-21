# Obsidian + Quartz Setup

## Open as Obsidian vault

1. Open Obsidian.
2. Choose Open folder as vault.
3. Select this folder: `portfolio-quartz/content`.
4. Edit notes in Obsidian. Quartz will publish them as your website.

## Run Quartz locally

From `portfolio-quartz`:

```bash
npm install
npx quartz build --serve
```

## Build for deployment

```bash
npx quartz build
```

Generated static site output is in the `public` folder.
