import type { SidebarsConfig } from '@docusaurus/plugin-content-docs';

const sidebars: SidebarsConfig = {
  docs: [
    {
      type: 'category',
      label: 'Get Started',
      collapsed: false,
      items: [
        'index',
        'installation',
        'getting-started',
        'user-guide/model-types',
      ],
    },
    {
      type: 'category',
      label: 'Guides',
      collapsed: false,
      items: [
        {
          type: 'category',
          label: 'Tabular Regression',
          items: [
            'user-guide/python-estimator',
            'user-guide/parameters',
            'objectives',
            'constraints',
            'spatial_modeling',
            'feature_schema',
            'sparse_features',
            'shap',
            'model_artifact',
          ],
        },
        {
          type: 'category',
          label: 'Forecasting',
          items: [
            'forecasting',
            'forecasting_api',
            'forecasting_backtesting',
            'forecasting_lag_features',
            'forecasting_artifacts',
            'forecasting_cli',
            'forecasting_examples',
          ],
        },
        {
          type: 'category',
          label: 'Model Guides',
          items: [
            'user-guide/forecasting-models/index',
            'forecasting_models',
            'user-guide/forecasting-models/naive-seasonal',
            'user-guide/forecasting-models/theta',
            'user-guide/forecasting-models/ets',
            'user-guide/forecasting-models/arima',
            'user-guide/forecasting-models/arima-examples',
            'user-guide/forecasting-models/kalman',
            'user-guide/forecasting-models/kriging',
            'user-guide/forecasting-models/cartoboost-lag',
            'user-guide/forecasting-models/ensembles',
          ],
        },
        'graph-features',
        'neural-features',
        'evaluation_protocol',
        'general_utilities',
        'user-guide/cli',
      ],
    },
    {
      type: 'category',
      label: 'Reference',
      collapsed: false,
      items: ['reference/python-api', 'reference/cli', 'feature_catalog'],
    },
    {
      type: 'category',
      label: 'Benchmarks',
      collapsed: false,
      items: [
        'benchmarks/index',
        'benchmarks/fair-benchmarking',
        'benchmarks/model-suite',
        'benchmarks/nyc-taxi',
        'benchmarks/forecasting',
        'benchmarks/taxi-zone',
        'benchmarks/neural-embedding-strategy',
        'benchmarks/neural-embedding-benchmark-latest',
      ],
    },
  ],
};

export default sidebars;
