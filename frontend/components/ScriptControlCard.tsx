import React, { useState, useCallback } from 'react';
import type { Script } from '../types';
import { Status } from '../types';
import OutputDisplay from './OutputDisplay';
import PlayIcon from './icons/PlayIcon';
import SpinnerIcon from './icons/SpinnerIcon';
import CheckCircleIcon from './icons/CheckCircleIcon';
import XCircleIcon from './icons/XCircleIcon';
import ConfirmationDialog from './ConfirmationDialog';

interface ScriptControlCardProps {
  script: Script;
}

const ScriptControlCard: React.FC<ScriptControlCardProps> = ({ script }) => {
  const [status, setStatus] = useState<Status>(Status.Idle);
  const [output, setOutput] = useState<string>('');
  const [lastRun, setLastRun] = useState<string | null>(null);
  const [isConfirming, setIsConfirming] = useState(false);

  const executeScript = useCallback(async () => {
    setStatus(Status.Running);
    const startedAt = new Date();
    setOutput(`[${startedAt.toLocaleTimeString()}] POST ${script.endpoint}\n`);
    try {
      const res = await fetch(script.endpoint, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: script.payload ? JSON.stringify(script.payload) : undefined
      });
      const text = await res.text();
      let body: any;
      try { body = JSON.parse(text); } catch { body = { raw: text }; }

      if (!res.ok) {
        setStatus(Status.Error);
        setOutput(prev => prev + `Request failed (${res.status}).\n${text}`);
      } else {
        const aggregated = body?.output || body?.stdout || text;
        setOutput(prev => prev + (aggregated || '\n[no output]'));
        setStatus((body?.returncode ?? 0) === 0 ? Status.Success : Status.Error);
      }
    } catch (e: any) {
      setStatus(Status.Error);
      setOutput(prev => prev + `\n[ERROR] ${e?.message || 'Network error'}`);
    }
    setLastRun(new Date().toLocaleString());
  }, [script]);
  
  const handleRunClick = () => {
    setIsConfirming(true);
  };

  const handleConfirm = () => {
    setIsConfirming(false);
    executeScript();
  };

  const handleCancel = () => {
    setIsConfirming(false);
  };

  const getStatusIndicator = () => {
    switch (status) {
      case Status.Running:
        return <div className="text-blue-400 flex items-center"><SpinnerIcon className="w-4 h-4 mr-2" /> Running...</div>;
      case Status.Success:
        return <div className="text-green-400 flex items-center"><CheckCircleIcon className="w-4 h-4 mr-2" /> Success</div>;
      case Status.Error:
        return <div className="text-red-400 flex items-center"><XCircleIcon className="w-4 h-4 mr-2" /> Error</div>;
      default:
        return <div className="text-gray-400">Idle</div>;
    }
  };

  return (
    <>
      <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 shadow-lg transform hover:scale-[1.02] transition-transform duration-300 flex flex-col">
        <div className="flex-grow">
          <h2 className="text-2xl font-bold text-white">{script.name}</h2>
          <p className="text-gray-400 mt-2 mb-4 h-12">{script.description}</p>
          <div className="text-xs text-gray-500 mb-4">
            <p>Last run: {lastRun || 'Never'}</p>
            <p>Status: {getStatusIndicator()}</p>
          </div>
        </div>
        
        <div className="flex-shrink-0">
          <button
            onClick={handleRunClick}
            disabled={status === Status.Running}
            className="w-full flex items-center justify-center px-4 py-3 bg-cyan-600 text-white font-bold rounded-md hover:bg-cyan-500 disabled:bg-gray-600 disabled:cursor-not-allowed transition-colors duration-300 focus:outline-none focus:ring-2 focus:ring-cyan-500 focus:ring-offset-2 focus:ring-offset-gray-800"
          >
            {status === Status.Running ? (
              <SpinnerIcon className="w-5 h-5 mr-2" />
            ) : (
              <PlayIcon className="w-5 h-5 mr-2" />
            )}
            {status === Status.Running ? 'Running...' : 'Run Script'}
          </button>
        </div>
        
        {output && <OutputDisplay output={output} />}
      </div>
      <ConfirmationDialog 
        isOpen={isConfirming}
        onConfirm={handleConfirm}
        onCancel={handleCancel}
        scriptName={script.name}
      />
    </>
  );
};

export default ScriptControlCard;
