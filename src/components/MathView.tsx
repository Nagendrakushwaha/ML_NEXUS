import React, { useEffect, useRef } from 'react';
import katex from 'katex';
import 'katex/dist/katex.min.css';

interface MathViewProps {
  formula: string;
}

export const MathView: React.FC<MathViewProps> = ({ formula }) => {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (containerRef.current) {
      katex.render(formula, containerRef.current, {
        throwOnError: false,
        displayMode: true
      });
    }
  }, [formula]);

  return <div ref={containerRef} className="py-4 overflow-x-auto text-lg" />;
};
