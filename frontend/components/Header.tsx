
import React from 'react';

const Header: React.FC = () => {
  return (
    <header className="bg-gray-800/50 backdrop-blur-sm border-b border-gray-700 sticky top-0 z-10">
      <div className="container mx-auto px-4 py-3 flex justify-between items-center">
        <div className="flex items-center space-x-3">
          <svg 
            className="w-8 h-8 text-cyan-500" 
            viewBox="0 0 24 24" 
            strokeWidth="1.5" 
            stroke="currentColor" 
            fill="none" 
            strokeLinecap="round" 
            strokeLinejoin="round"
          >
            <path stroke="none" d="M0 0h24v24H0z" fill="none"/>
            <path d="M10.003 16.005a2.5 2.5 0 1 0 4 0a2.5 2.5 0 1 0 -4 0" />
            <path d="M12.003 3v5.5" />
            <path d="M12.003 18.5v2.5" />
            <path d="M15.913 6.42l3.434 -2.93" />
            <path d="M4.656 17.35l3.434 -2.93" />
            <path d="M19.347 17.35l-3.434 -2.93" />
            <path d="M8.09 6.42l-3.434 -2.93" />
            <path d="M3 12.005h5.5" />
            <path d="M15.5 12.005h5.5" />
          </svg>
          <span className="text-xl font-bold text-white">FootyTips Cloud Runner</span>
        </div>
        <div className="flex items-center space-x-2">
            <div className="w-3 h-3 bg-green-500 rounded-full animate-pulse"></div>
            <span className="text-sm text-green-400">Server Online</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
