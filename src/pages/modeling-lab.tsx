import BrowserOnly from '@docusaurus/BrowserOnly';
import Layout from '@theme/Layout';

import ModelingLabClient from '../components/ModelingLabClient';

export default function ModelingLab(): React.ReactElement {
  return (
    <Layout
      title="Modeling Lab"
      description="Run CartoBoost forecasting, regression, graph, and neural modeling in the browser with WebAssembly"
    >
      <BrowserOnly fallback={<main />}>{() => <ModelingLabClient />}</BrowserOnly>
    </Layout>
  );
}
