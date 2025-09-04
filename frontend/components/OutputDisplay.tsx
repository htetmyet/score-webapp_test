
import React from 'react';

interface OutputDisplayProps {
  output: string;
}

const OutputDisplay: React.FC<OutputDisplayProps> = ({ output }) => {
  return (
    <div className="mt-4">
      <h4 className="text-sm font-semibold text-gray-300 mb-2">Output Log:</h4>
      <pre className="bg-gray-900 text-white text-xs p-4 rounded-md overflow-x-auto max-h-60 custom-scrollbar">
        <code>{output}</code>
      </pre>
      <style>{`
        .custom-scrollbar::-webkit-scrollbar {
          width: 8px;
          height: 8px;
        }
        .custom-scrollbar::-webkit-scrollbar-track {
          background: #2D2D2D;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb {
          background: #424242;
          border-radius: 4px;
        }
        .custom-scrollbar::-webkit-scrollbar-thumb:hover {
          background: #555;
        }
      `}</style>
    </div>
  );
};

export default OutputDisplay;
