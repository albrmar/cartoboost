import React from 'react';
import DocItemContent from '@theme-original/DocItem/Content';
import type DocItemContentType from '@theme/DocItem/Content';

import AgentCopyControls from '../../../components/AgentCopyControls';

type Props = React.ComponentProps<typeof DocItemContentType>;

export default function DocItemContentWrapper(props: Props): React.JSX.Element {
  return (
    <>
      <AgentCopyControls />
      <DocItemContent {...props} />
    </>
  );
}
