import BrowserOnly from '@docusaurus/BrowserOnly';
import Layout from '@theme/Layout';

import ForecastLabClient from '../components/ForecastLabClient';

export default function ForecastLab(): React.ReactElement {
  return (
    <Layout
      title="Forecast Lab"
      description="Run CartoBoost forecasting in the browser with WebAssembly"
    >
      <BrowserOnly fallback={<main />}>{() => <ForecastLabClient />}</BrowserOnly>
    </Layout>
  );
}
