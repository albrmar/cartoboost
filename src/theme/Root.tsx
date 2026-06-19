import React from 'react';

import ImageZoom from '../components/ImageZoom';

type RootProps = {
  children: React.ReactNode;
};

export default function Root({ children }: RootProps): React.JSX.Element {
  return (
    <>
      {children}
      <ImageZoom />
    </>
  );
}
