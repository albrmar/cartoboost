import React, { useEffect, useState } from 'react';

type ZoomImage = {
  alt: string;
  src: string;
};

function isZoomableImage(target: EventTarget | null): target is HTMLImageElement {
  if (!(target instanceof HTMLImageElement)) {
    return false;
  }
  if (!target.closest('main')) {
    return false;
  }
  if (target.closest('a, button, .navbar, footer')) {
    return false;
  }
  return Boolean(target.currentSrc || target.src);
}

function zoomImageFromElement(image: HTMLImageElement): ZoomImage {
  return {
    alt: image.alt || 'Documentation image',
    src: image.currentSrc || image.src,
  };
}

export default function ImageZoom(): React.JSX.Element | null {
  const [zoomImage, setZoomImage] = useState<ZoomImage | null>(null);

  useEffect(() => {
    function prepareImages(): void {
      const images = Array.from(document.querySelectorAll<HTMLImageElement>('main img'));
      for (const image of images) {
        if (!image.closest('a, button, .navbar, footer')) {
          image.classList.add('zoomable-image');
          image.tabIndex = 0;
          image.setAttribute('role', 'button');
          image.setAttribute('title', 'Click to zoom');
        }
      }
    }

    prepareImages();
    const observer = new MutationObserver(prepareImages);
    observer.observe(document.body, { childList: true, subtree: true });

    function onClick(event: MouseEvent): void {
      if (isZoomableImage(event.target)) {
        setZoomImage(zoomImageFromElement(event.target));
      }
    }

    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        setZoomImage(null);
        return;
      }
      if ((event.key === 'Enter' || event.key === ' ') && isZoomableImage(event.target)) {
        event.preventDefault();
        setZoomImage(zoomImageFromElement(event.target));
      }
    }

    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      observer.disconnect();
      document.removeEventListener('click', onClick);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, []);

  if (zoomImage === null) {
    return null;
  }

  return (
    <div className="image-zoom-backdrop" role="presentation" onClick={() => setZoomImage(null)}>
      <div className="image-zoom-frame" role="dialog" aria-modal="true" aria-label={zoomImage.alt}>
        <button className="image-zoom-close" type="button" onClick={() => setZoomImage(null)}>
          Close
        </button>
        <img src={zoomImage.src} alt={zoomImage.alt} onClick={(event) => event.stopPropagation()} />
        {zoomImage.alt ? <p>{zoomImage.alt}</p> : null}
      </div>
    </div>
  );
}
