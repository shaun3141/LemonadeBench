import { useState, useEffect, useCallback } from 'react';
import { getHealth } from '@/api';

const HEALTH_CHECK_INTERVAL = 5000; // 5 seconds

/**
 * Hook for managing server connection status
 */
export function useServerConnection() {
  const [isConnected, setIsConnected] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const checkConnection = useCallback(async () => {
    try {
      await getHealth();
      setIsConnected(true);
      setError(null);
    } catch {
      setIsConnected(false);
      setError(
        'Cannot connect to server. Make sure the LemonadeBench server is running on port 8000.'
      );
    }
  }, []);

  useEffect(() => {
    checkConnection();
    const interval = setInterval(checkConnection, HEALTH_CHECK_INTERVAL);
    return () => clearInterval(interval);
  }, [checkConnection]);

  return {
    isConnected,
    connectionError: error,
    checkConnection,
  };
}

