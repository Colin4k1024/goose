export function getRsiArgs(request: unknown): string[] {
  if (!request || typeof request !== 'object' || !('action' in request)) {
    throw new Error('Invalid RSI command');
  }

  switch (request.action) {
    case 'init':
    case 'status':
    case 'history':
    case 'evolve':
      return ['rsi', request.action];
    case 'rollback':
      if (
        !('version' in request) ||
        typeof request.version !== 'number' ||
        !Number.isSafeInteger(request.version) ||
        request.version < 1
      ) {
        throw new Error('A positive backup version is required');
      }
      return ['rsi', 'rollback', String(request.version)];
    default:
      throw new Error('Unsupported RSI command');
  }
}
