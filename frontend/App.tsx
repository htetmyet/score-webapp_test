
import React from 'react';
import Header from './components/Header';
import ScriptControlCard from './components/ScriptControlCard';
import type { Script } from './types';

const App: React.FC = () => {

  const scripts: Script[] = [
    {
      id: 'data-process',
      name: 'Fetch Latest Performance & Scraping',
      description: 'Fetches latest performance data and updates team/referee scrapings.',
      endpoint: '/api/run/data-process',
      steps: ['z_latest_performance.py', 'z_latest_team_scr_ref.py']
    },
    {
      id: 'prepare-data',
      name: 'Data Augmentation',
      description: 'Runs data augemntation process to build cleaned dataset.',
      endpoint: '/api/run/prepare-data',
      steps: ['adjust_team_perform.py','set_train_data.py','ml_models/data-eng.py']
    },
    {
      id: 'train-model',
      name: 'Train All Models',
      description: 'Runs full pipeline (prep + result, AH, and goals models).',
      endpoint: '/api/run/train-model',
      steps: [
        'adjust_team_perform.py', 
        'set_train_data.py', 
        'ml_models/data-eng.py',
        'ml_models/train_model_new.py', 
        'ml_models/train_AH_model.py', 
        'ml_models/train_goals_model.py'
      ]
    },
    {
      id: 'train-result-only',
      name: 'Train Result Model Only',
      description: 'Trains just the match result model (assumes dataset prepared).',
      endpoint: '/api/run/train-selected',
      steps: ['ml_models/train_model_new.py'],
      payload: { models: ['result'] }
    },
    {
      id: 'train-ah-only',
      name: 'Train AH Model Only',
      description: 'Trains only the Asian Handicap model (assumes dataset prepared).',
      endpoint: '/api/run/train-selected',
      steps: ['ml_models/train_AH_model.py'],
      payload: { models: ['ah'] }
    },
    {
      id: 'train-goals-only',
      name: 'Train Goals Model Only',
      description: 'Trains only the Over/Under goals model (assumes dataset prepared).',
      endpoint: '/api/run/train-selected',
      steps: ['ml_models/train_goals_model.py'],
      payload: { models: ['goals'] }
    },
    {
      id: 'predict-res',
      name: 'Generate Predictions',
      description: 'Uses the trained models to generate predictions for upcoming fixtures.',
      endpoint: '/api/run/predict-res',
      steps: [
        'ml_models/predict_fixtures.py', 
        'ml_models/predict_AH_model.py', 
        'ml_models/predict_goals.py'
      ]
    },
    {
      id: 'send-telegram',
      name: 'Send Predictions As Telegram Message',
      description: 'Sends a summary of the predictions to a Telegram channel.',
      endpoint: '/api/run/send-telegram'
    }
  ];

  return (
    <div className="min-h-screen bg-gray-900 font-sans">
      <Header />
      <main className="container mx-auto px-4 py-8">
        <h1 className="text-3xl md:text-4xl font-bold text-white mb-2">Lemme Cook!</h1>
        <p className="text-gray-400 mb-8">Trigger and monitor the predictions.</p>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-2 gap-6">
          {scripts.map((script) => (
            <ScriptControlCard key={script.id} script={script} />
          ))}
        </div>
      </main>
      <footer className="text-center py-6 text-gray-600">
        <p>Footy At Ease &copy; 2025</p>
      </footer>
    </div>
  );
};

export default App;
