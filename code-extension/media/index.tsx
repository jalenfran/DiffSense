import React from 'react';
import { createRoot } from 'react-dom/client';
import ChatApp from './ChatApp';

// Acquire VS Code API
declare global {
  interface Window {
    acquireVsCodeApi: () => any;
  }
}

const vscode = window.acquireVsCodeApi();

const container = document.getElementById('root');
if (container) {
  const root = createRoot(container);
  root.render(<ChatApp vscode={vscode} />);
}
