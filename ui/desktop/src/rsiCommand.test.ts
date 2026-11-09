import { describe, expect, it } from 'vitest';
import { getRsiArgs } from './rsiCommand';

describe('getRsiArgs', () => {
  it('maps supported actions to fixed CLI arguments', () => {
    expect(getRsiArgs({ action: 'status' })).toEqual(['rsi', 'status']);
    expect(getRsiArgs({ action: 'rollback', version: 3 })).toEqual(['rsi', 'rollback', '3']);
  });

  it('rejects arbitrary commands and invalid rollback versions', () => {
    expect(() => getRsiArgs({ action: 'run' })).toThrow('Unsupported RSI command');
    expect(() => getRsiArgs({ action: 'rollback', version: '3; rm -rf /' })).toThrow();
    expect(() => getRsiArgs({ action: 'rollback', version: 0 })).toThrow();
  });
});
