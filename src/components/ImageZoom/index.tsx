import React, { useCallback, useEffect, useRef, useState } from 'react';

type ZoomTarget = {
  alt: string;
  content: string;
  kind: 'image' | 'svg';
};

type PanState = {
  originX: number;
  originY: number;
  pointerId: number;
  startX: number;
  startY: number;
};

const MIN_ZOOM = 0.5;
const MAX_ZOOM = 8;
const ZOOM_STEP = 0.35;

function clampZoom(value: number): number {
  return Math.min(MAX_ZOOM, Math.max(MIN_ZOOM, value));
}

function isBlockedTarget(element: Element): boolean {
  return Boolean(element.closest('a, button, input, textarea, select, .navbar, footer, [data-image-zoom-ui]'));
}

function isMermaidSvg(element: SVGSVGElement): boolean {
  return Boolean(
    element.closest('.docusaurus-mermaid-container, .mermaid, [class*="mermaid"]') &&
      element.closest('main') &&
      !isBlockedTarget(element),
  );
}

function findZoomableElement(
  target: EventTarget | null,
  event?: Event,
): HTMLImageElement | SVGSVGElement | null {
  if (!(target instanceof Element)) {
    return null;
  }

  const image = target.closest('main img');
  if (image instanceof HTMLImageElement && !isBlockedTarget(image) && (image.currentSrc || image.src)) {
    return image;
  }

  const mermaidSvg = target.closest(
    '.docusaurus-mermaid-container svg, .mermaid svg, [class*="mermaid"] svg',
  );
  if (mermaidSvg instanceof SVGSVGElement && mermaidSvg.closest('main') && !isBlockedTarget(mermaidSvg)) {
    return mermaidSvg;
  }

  for (const item of event?.composedPath() ?? []) {
    if (item instanceof SVGSVGElement && isMermaidSvg(item)) {
      return item;
    }
    if (item instanceof SVGElement && item.ownerSVGElement && isMermaidSvg(item.ownerSVGElement)) {
      return item.ownerSVGElement;
    }
  }

  return null;
}

function zoomTargetFromElement(element: HTMLImageElement | SVGSVGElement): ZoomTarget {
  if (element instanceof HTMLImageElement) {
    return {
      alt: element.alt || 'Documentation image',
      content: element.currentSrc || element.src,
      kind: 'image',
    };
  }

  const clone = element.cloneNode(true) as SVGSVGElement;
  clone.removeAttribute('height');
  clone.removeAttribute('width');
  clone.setAttribute('preserveAspectRatio', 'xMidYMid meet');

  return {
    alt:
      element.getAttribute('aria-label') ||
      element.closest('[aria-label]')?.getAttribute('aria-label') ||
      'Documentation diagram',
    content: clone.outerHTML,
    kind: 'svg',
  };
}

export default function ImageZoom(): React.JSX.Element | null {
  const [zoomTarget, setZoomTarget] = useState<ZoomTarget | null>(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState({ x: 0, y: 0 });
  const panRef = useRef<PanState | null>(null);

  const resetView = useCallback(() => {
    setZoom(1);
    setOffset({ x: 0, y: 0 });
  }, []);

  const close = useCallback(() => {
    setZoomTarget(null);
    resetView();
  }, [resetView]);

  const open = useCallback(
    (element: HTMLImageElement | SVGSVGElement) => {
      setZoomTarget(zoomTargetFromElement(element));
      resetView();
    },
    [resetView],
  );

  useEffect(() => {
    function prepareZoomables(): void {
      const images = Array.from(document.querySelectorAll<HTMLImageElement>('main img'));
      for (const image of images) {
        if (!image.closest('a, button, .navbar, footer')) {
          image.classList.add('zoomable-image');
          image.tabIndex = 0;
          image.setAttribute('role', 'button');
          image.setAttribute('title', 'Click to zoom');
        }
      }

      const diagrams = Array.from(
        document.querySelectorAll<SVGSVGElement>(
          'main .docusaurus-mermaid-container svg, main .mermaid svg, main [class*="mermaid"] svg',
        ),
      );
      for (const diagram of diagrams) {
        if (!diagram.closest('a, button, .navbar, footer')) {
          diagram.classList.add('zoomable-diagram');
          diagram.tabIndex = 0;
          diagram.setAttribute('role', 'button');
          diagram.setAttribute('title', 'Click to zoom');
        }
      }
    }

    prepareZoomables();
    const observer = new MutationObserver(prepareZoomables);
    observer.observe(document.body, { childList: true, subtree: true });

    function onClick(event: MouseEvent): void {
      const zoomable = findZoomableElement(event.target, event);
      if (zoomable) {
        event.preventDefault();
        open(zoomable);
      }
    }

    function onKeyDown(event: KeyboardEvent): void {
      if (event.key === 'Escape') {
        close();
        return;
      }

      const zoomable = findZoomableElement(event.target, event);
      if ((event.key === 'Enter' || event.key === ' ') && zoomable) {
        event.preventDefault();
        open(zoomable);
      }
    }

    document.addEventListener('click', onClick);
    document.addEventListener('keydown', onKeyDown);
    return () => {
      observer.disconnect();
      document.removeEventListener('click', onClick);
      document.removeEventListener('keydown', onKeyDown);
    };
  }, [close, open]);

  useEffect(() => {
    if (zoomTarget === null) {
      return undefined;
    }

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    return () => {
      document.body.style.overflow = previousOverflow;
    };
  }, [zoomTarget]);

  if (zoomTarget === null) {
    return null;
  }

  function updateZoom(delta: number): void {
    setZoom((value) => clampZoom(value + delta));
  }

  function onWheel(event: React.WheelEvent<HTMLDivElement>): void {
    event.preventDefault();
    updateZoom(event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP);
  }

  function onPointerDown(event: React.PointerEvent<HTMLDivElement>): void {
    event.currentTarget.setPointerCapture(event.pointerId);
    panRef.current = {
      originX: offset.x,
      originY: offset.y,
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
    };
  }

  function onPointerMove(event: React.PointerEvent<HTMLDivElement>): void {
    const pan = panRef.current;
    if (!pan || pan.pointerId !== event.pointerId) {
      return;
    }
    setOffset({
      x: pan.originX + event.clientX - pan.startX,
      y: pan.originY + event.clientY - pan.startY,
    });
  }

  function onPointerUp(event: React.PointerEvent<HTMLDivElement>): void {
    if (panRef.current?.pointerId === event.pointerId) {
      panRef.current = null;
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  }

  return (
    <div className="image-zoom-backdrop" role="presentation" onClick={close} data-image-zoom-ui>
      <div className="image-zoom-frame" role="dialog" aria-modal="true" aria-label={zoomTarget.alt}>
        <div className="image-zoom-toolbar" onClick={(event) => event.stopPropagation()} data-image-zoom-ui>
          <p>{zoomTarget.alt}</p>
          <button type="button" onClick={() => updateZoom(-ZOOM_STEP)} aria-label="Zoom out">
            -
          </button>
          <button type="button" onClick={resetView}>
            Reset
          </button>
          <button type="button" onClick={() => updateZoom(ZOOM_STEP)} aria-label="Zoom in">
            +
          </button>
          <button type="button" onClick={close}>
            Close
          </button>
        </div>
        <div
          className="image-zoom-viewport"
          onClick={(event) => event.stopPropagation()}
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={onPointerUp}
          onPointerCancel={onPointerUp}
          onWheel={onWheel}
        >
          <div
            className="image-zoom-canvas"
            style={{
              transform: `translate(${offset.x}px, ${offset.y}px) scale(${zoom})`,
            }}
          >
            {zoomTarget.kind === 'image' ? (
              <img src={zoomTarget.content} alt={zoomTarget.alt} draggable={false} />
            ) : (
              <div
                className="image-zoom-svg"
                dangerouslySetInnerHTML={{ __html: zoomTarget.content }}
              />
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
