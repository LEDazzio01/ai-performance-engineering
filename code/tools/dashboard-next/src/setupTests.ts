import '@testing-library/jest-dom';
import React from 'react';

// Some legacy components rely on the classic runtime; expose React globally for tests.
(globalThis as any).React = React;

// Recharts needs ResizeObserver in jsdom.
class ResizeObserver {
  observe() {}
  unobserve() {}
  disconnect() {}
}

(globalThis as any).ResizeObserver = ResizeObserver;
