
export enum Status {
  Idle = 'idle',
  Running = 'running',
  Success = 'success',
  Error = 'error'
}

export interface Script {
  id: string;
  name: string;
  description: string;
  endpoint: string;
  steps?: string[];
  payload?: any;
}
