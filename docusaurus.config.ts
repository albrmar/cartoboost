import fs from 'node:fs';
import path from 'node:path';
import type { Config, Plugin } from '@docusaurus/types';
import type { Options as ClassicOptions } from '@docusaurus/preset-classic';
import { themes as prismThemes } from 'prism-react-renderer';

function llmsTxtDocsPlugin(): Plugin {
  return {
    name: 'llms-txt-docs',
    postBuild({ outDir }) {
      const source = path.join(process.cwd(), 'docs', 'llms.txt');
      const target = path.join(outDir, 'docs', 'llms.txt');
      fs.mkdirSync(path.dirname(target), { recursive: true });
      fs.copyFileSync(source, target);
    },
  };
}

const config: Config = {
  title: 'CartoBoost',
  tagline: 'Temporal, spatial, geotemporal, and graph-aware regression',
  favicon: 'img/cartoboost-route-splitter-logo.svg',

  url: 'https://theculliganman.github.io',
  baseUrl: '/CartoBoost/',
  organizationName: 'TheCulliganMan',
  projectName: 'CartoBoost',
  trailingSlash: false,

  onBrokenLinks: 'throw',
  onDuplicateRoutes: 'throw',

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  markdown: {
    mermaid: true,
    hooks: {
      onBrokenMarkdownLinks: 'throw',
    },
  },

  presets: [
    [
      'classic',
      {
        docs: {
          path: 'docs',
          routeBasePath: 'docs',
          sidebarPath: './sidebars.ts',
          editUrl: 'https://github.com/TheCulliganMan/CartoBoost/edit/main/',
          showLastUpdateAuthor: true,
          showLastUpdateTime: true,
          exclude: ['README.md', 'llms.txt', '**/AGENTS.md', 'assets/**'],
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
        sitemap: {
          changefreq: 'weekly',
          priority: 0.7,
          ignorePatterns: ['/tags/**'],
          filename: 'sitemap.xml',
        },
      } satisfies ClassicOptions,
    ],
  ],

  themes: ['@docusaurus/theme-mermaid'],

  plugins: [
    llmsTxtDocsPlugin,
    [
      '@docusaurus/plugin-client-redirects',
      {
        redirects: [
          { from: '/installation', to: '/docs/installation' },
          { from: '/getting-started', to: '/docs/getting-started' },
          { from: '/feature_catalog', to: '/docs/feature_catalog' },
          { from: '/forecasting', to: '/docs/forecasting' },
          { from: '/benchmarks', to: '/docs/benchmarks' },
          { from: '/reference/python-api', to: '/docs/reference/python-api' },
          { from: '/reference/cli', to: '/docs/reference/cli' },
          { from: '/user-guide/model-types', to: '/docs/user-guide/model-types' },
          { from: '/user-guide/python-estimator', to: '/docs/user-guide/python-estimator' },
        ],
      },
    ],
  ],

  themeConfig: {
    image: 'img/social-card.svg',
    colorMode: {
      defaultMode: 'dark',
      disableSwitch: false,
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'CartoBoost',
      logo: {
        alt: 'CartoBoost',
        src: 'img/cartoboost-route-splitter-logo.svg',
      },
      items: [
        { to: '/docs/installation', label: 'Get Started', position: 'left' },
        { to: '/docs/user-guide/model-types', label: 'Guides', position: 'left' },
        { to: '/forecast-lab', label: 'Forecast Lab', position: 'left' },
        { to: '/docs/reference/python-api', label: 'Reference', position: 'left' },
        { to: '/docs/benchmarks', label: 'Benchmarks', position: 'left' },
        {
          href: 'https://github.com/TheCulliganMan/CartoBoost',
          label: 'GitHub',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Start',
          items: [
            { label: 'Install', to: '/docs/installation' },
            { label: 'Getting Started', to: '/docs/getting-started' },
            { label: 'Choose A Model', to: '/docs/user-guide/model-types' },
          ],
        },
        {
          title: 'Deep Dives',
          items: [
            { label: 'Forecasting', to: '/docs/forecasting' },
            { label: 'Graph Features', to: '/docs/graph-features' },
            { label: 'Neural Features', to: '/docs/neural-features' },
          ],
        },
        {
          title: 'Evidence',
          items: [
            { label: 'Benchmarks', to: '/docs/benchmarks' },
            { label: 'Feature Catalog', to: '/docs/feature_catalog' },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} CartoBoost contributors.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['bash', 'toml', 'rust'],
    },
  },
};

export default config;
