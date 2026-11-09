export type RsiCommand =
  { action: 'init' | 'status' | 'history' | 'evolve' } | { action: 'rollback'; version: number };
