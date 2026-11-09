import { fireEvent, render, screen, waitFor } from '@testing-library/react';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { IntlTestWrapper } from '../../i18n/test-utils';
import type { RsiCommand } from '../../types/rsi';
import RsiView from './RsiView';

describe('RsiView', () => {
  const runRsi = vi.fn(async (command: RsiCommand) => {
    switch (command.action) {
      case 'status':
        return 'Cycle count: 2';
      case 'history':
        return 'Cycle 2: accepted';
      case 'evolve':
        return 'cycle 3: solidified=true';
      default:
        return 'done';
    }
  });

  beforeEach(() => {
    vi.clearAllMocks();
    window.electron.runRsi = runRsi;
  });

  it('shows the current RSI status and history', async () => {
    render(<RsiView />, { wrapper: IntlTestWrapper });

    expect(await screen.findByText('Cycle count: 2')).toBeInTheDocument();
    expect(screen.getByText('Cycle 2: accepted')).toBeInTheDocument();
    expect(runRsi).toHaveBeenCalledWith({ action: 'status' });
    expect(runRsi).toHaveBeenCalledWith({ action: 'history' });
  });

  it('confirms a cycle before starting the real RSI command', async () => {
    vi.mocked(window.electron.showMessageBox).mockResolvedValue({ response: 1 });
    render(<RsiView />, { wrapper: IntlTestWrapper });
    await screen.findByText('Cycle count: 2');

    fireEvent.click(screen.getByRole('button', { name: 'Run one cycle' }));

    await waitFor(() => expect(runRsi).toHaveBeenCalledWith({ action: 'evolve' }));
    expect(await screen.findByText('cycle 3: solidified=true')).toBeInTheDocument();
  });
});
