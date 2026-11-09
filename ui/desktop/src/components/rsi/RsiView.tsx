import { useCallback, useEffect, useState } from 'react';
import { Play, RefreshCw, RotateCcw, Undo2 } from 'lucide-react';
import { defineMessages, useIntl } from '../../i18n';
import type { RsiCommand } from '../../types/rsi';
import { errorMessage } from '../../utils/conversionUtils';
import { MainPanelLayout } from '../Layout/MainPanelLayout';
import { Button } from '../ui/button';
import { ScrollArea } from '../ui/scroll-area';

const i18n = defineMessages({
  title: { id: 'rsi.title', defaultMessage: 'RSI Evolution' },
  description: {
    id: 'rsi.description',
    defaultMessage:
      'Evaluate recipe changes against fixed training and holdout tasks before accepting them.',
  },
  refresh: { id: 'rsi.refresh', defaultMessage: 'Refresh' },
  initialize: { id: 'rsi.initialize', defaultMessage: 'Install sample benchmark' },
  evolve: { id: 'rsi.evolve', defaultMessage: 'Run one cycle' },
  running: { id: 'rsi.running', defaultMessage: 'Running…' },
  status: { id: 'rsi.status', defaultMessage: 'Status' },
  history: { id: 'rsi.history', defaultMessage: 'Recent events' },
  loading: { id: 'rsi.loading', defaultMessage: 'Loading…' },
  rollback: { id: 'rsi.rollback', defaultMessage: 'Restore backup' },
  backupVersion: { id: 'rsi.backupVersion', defaultMessage: 'Backup version' },
  rollbackDescription: {
    id: 'rsi.rollbackDescription',
    defaultMessage: 'Enter a backup version shown in Status to restore the managed recipe.',
  },
  confirmEvolve: {
    id: 'rsi.confirmEvolve',
    defaultMessage:
      'Run one RSI evolution cycle? This calls your configured model, evaluates the candidate, and may change the managed recipe.',
  },
  confirmRollback: {
    id: 'rsi.confirmRollback',
    defaultMessage: 'Restore backup version {version}? This replaces the current managed recipe.',
  },
  cancel: { id: 'rsi.cancel', defaultMessage: 'Cancel' },
  run: { id: 'rsi.run', defaultMessage: 'Run cycle' },
  restore: { id: 'rsi.restore', defaultMessage: 'Restore' },
});

export default function RsiView() {
  const intl = useIntl();
  const [status, setStatus] = useState('');
  const [history, setHistory] = useState('');
  const [backupVersion, setBackupVersion] = useState('');
  const [loading, setLoading] = useState(false);
  const [running, setRunning] = useState(false);
  const [result, setResult] = useState('');
  const [error, setError] = useState('');

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      const [nextStatus, nextHistory] = await Promise.all([
        window.electron.runRsi({ action: 'status' }),
        window.electron.runRsi({ action: 'history' }),
      ]);
      setStatus(nextStatus);
      setHistory(nextHistory);
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const runCommand = async (command: RsiCommand) => {
    setRunning(true);
    setResult('');
    setError('');
    try {
      setResult(await window.electron.runRsi(command));
    } catch (cause) {
      setError(errorMessage(cause));
    } finally {
      await refresh();
      setRunning(false);
    }
  };

  const confirmEvolve = async () => {
    const answer = await window.electron.showMessageBox({
      type: 'warning',
      title: intl.formatMessage(i18n.evolve),
      message: intl.formatMessage(i18n.confirmEvolve),
      buttons: [intl.formatMessage(i18n.cancel), intl.formatMessage(i18n.run)],
      defaultId: 0,
    });
    if (answer.response === 1) void runCommand({ action: 'evolve' });
  };

  const version = Number(backupVersion);
  const validVersion = Number.isSafeInteger(version) && version > 0;
  const confirmRollback = async () => {
    if (!validVersion) return;
    const answer = await window.electron.showMessageBox({
      type: 'warning',
      title: intl.formatMessage(i18n.rollback),
      message: intl.formatMessage(i18n.confirmRollback, { version }),
      buttons: [intl.formatMessage(i18n.cancel), intl.formatMessage(i18n.restore)],
      defaultId: 0,
    });
    if (answer.response === 1) void runCommand({ action: 'rollback', version });
  };

  return (
    <MainPanelLayout>
      <div className="flex min-h-0 flex-1 flex-col">
        <div className="px-8 pb-6 pt-16">
          <h1 className="text-4xl font-light">{intl.formatMessage(i18n.title)}</h1>
          <p className="mt-2 text-sm text-text-secondary">{intl.formatMessage(i18n.description)}</p>
          <div className="mt-5 flex flex-wrap gap-2">
            <Button
              variant="outline"
              size="sm"
              onClick={() => void refresh()}
              disabled={loading || running}
            >
              <RefreshCw className="mr-2 h-4 w-4" />
              {intl.formatMessage(i18n.refresh)}
            </Button>
            <Button
              variant="outline"
              size="sm"
              onClick={() => void runCommand({ action: 'init' })}
              disabled={running}
            >
              <RotateCcw className="mr-2 h-4 w-4" />
              {intl.formatMessage(i18n.initialize)}
            </Button>
            <Button size="sm" onClick={() => void confirmEvolve()} disabled={running || loading}>
              <Play className="mr-2 h-4 w-4" />
              {running ? intl.formatMessage(i18n.running) : intl.formatMessage(i18n.evolve)}
            </Button>
          </div>
        </div>

        <ScrollArea className="min-h-0 flex-1 px-8 pb-8">
          <div className="max-w-4xl space-y-6">
            {error && (
              <div
                role="alert"
                className="rounded-lg bg-red-500/10 p-4 text-sm text-red-600 dark:text-red-400"
              >
                {error}
              </div>
            )}
            {result && (
              <div
                role="status"
                className="whitespace-pre-wrap rounded-lg bg-green-500/10 p-4 text-sm text-text-primary"
              >
                {result}
              </div>
            )}

            <section className="rounded-xl bg-background-secondary p-5">
              <h2 className="mb-3 text-lg font-medium">{intl.formatMessage(i18n.status)}</h2>
              <pre className="whitespace-pre-wrap break-words font-mono text-sm text-text-secondary">
                {status || (loading ? intl.formatMessage(i18n.loading) : '')}
              </pre>
            </section>

            <section className="rounded-xl bg-background-secondary p-5">
              <h2 className="mb-3 text-lg font-medium">{intl.formatMessage(i18n.history)}</h2>
              <pre className="whitespace-pre-wrap break-words font-mono text-sm text-text-secondary">
                {history || (loading ? intl.formatMessage(i18n.loading) : '')}
              </pre>
            </section>

            <section className="rounded-xl bg-background-secondary p-5">
              <h2 className="mb-1 text-lg font-medium">{intl.formatMessage(i18n.rollback)}</h2>
              <p className="mb-3 text-sm text-text-secondary">
                {intl.formatMessage(i18n.rollbackDescription)}
              </p>
              <div className="flex items-center gap-2">
                <input
                  type="number"
                  min="1"
                  step="1"
                  value={backupVersion}
                  onChange={(event) => setBackupVersion(event.target.value)}
                  placeholder={intl.formatMessage(i18n.backupVersion)}
                  aria-label={intl.formatMessage(i18n.backupVersion)}
                  className="h-9 w-40 rounded-md border border-border bg-background-primary px-3 text-sm"
                />
                <Button
                  variant="outline"
                  size="sm"
                  disabled={!validVersion || running}
                  onClick={() => void confirmRollback()}
                >
                  <Undo2 className="mr-2 h-4 w-4" />
                  {intl.formatMessage(i18n.restore)}
                </Button>
              </div>
            </section>
          </div>
        </ScrollArea>
      </div>
    </MainPanelLayout>
  );
}
