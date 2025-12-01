'use client';

import { useState } from 'react';
import { motion } from 'framer-motion';
import { toast } from 'sonner';
import { MicrophoneIcon } from '@phosphor-icons/react/dist/ssr';

interface WelcomeViewProps {
  onStartCall: (playerName: string) => void;
}

export const WelcomeView = ({ onStartCall }: WelcomeViewProps) => {
  const [playerName, setPlayerName] = useState('');
  const [isConnecting, setIsConnecting] = useState(false);

  const handleStart = async () => {
    if (!playerName.trim()) {
      toast.error('Please enter your stage name');
      return;
    }

    setIsConnecting(true);
    toast.loading('Connecting to the stage...', { id: 'connecting' });

    try {
      await new Promise((resolve) => setTimeout(resolve, 500)); // Small delay for UX
      onStartCall(playerName.trim());
      toast.success('Connected!', { id: 'connecting' });
    } catch (error) {
      toast.error('Connection failed, please try again.', { id: 'connecting' });
      setIsConnecting(false);
    }
  };

  return (
    <div className="relative min-h-screen w-full overflow-hidden">
      {/* Animated gradient background - Slate/Dark Gray to Black */}
      <div className="absolute inset-0 bg-gradient-to-br from-gray-900 via-black to-gray-900">
        {/* Subtle Orange/Gold accents */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(249,115,22,0.3),transparent_50%)]" /> {/* Orange-600 */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(251,191,36,0.2),transparent_50%)]" /> {/* Amber/Gold */}
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(249,115,22,0.1),transparent_70%)]" />
      </div>

      {/* Blurred colored blobs for stage lighting effect - Orange & Gold */}
      <div className="absolute top-0 left-0 h-96 w-96 rounded-full bg-orange-500/20 blur-3xl" />
      <div className="absolute bottom-0 right-0 h-96 w-96 rounded-full bg-yellow-500/20 blur-3xl" />

      {/* Main content */}
      <div className="relative z-10 flex min-h-screen items-center justify-center p-6">
        <motion.div
          initial={{ opacity: 0, scale: 0.95, y: 20 }}
          animate={{ opacity: 1, scale: 1, y: 0 }}
          transition={{ duration: 0.5, ease: 'easeOut' }}
          className="w-full max-w-md"
        >
          {/* Glassmorphism card */}
          <div className="relative overflow-hidden rounded-2xl border border-white/10 bg-white/5 backdrop-blur-xl shadow-2xl">
            {/* Inner glow */}
            <div className="absolute inset-0 bg-gradient-to-br from-white/10 to-transparent" />
            
            <div className="relative p-8 md:p-10">
              {/* Title and icon */}
              <div className="mb-8 text-center">
                <motion.div
                  initial={{ scale: 0 }}
                  animate={{ scale: 1 }}
                  transition={{ delay: 0.2, type: 'spring', stiffness: 200 }}
                  className="mb-4 flex justify-center"
                >
                  {/* Icon gradient changed to Orange/Red */}
                  <div className="flex h-16 w-16 items-center justify-center rounded-full bg-gradient-to-br from-orange-600 to-red-600 shadow-lg">
                    <MicrophoneIcon size={32} weight="fill" className="text-white" />
                  </div>
                </motion.div>
                
                <h1 className="mb-2 text-3xl font-bold text-white md:text-4xl">
                  Improv Battle
                </h1>
                <p className="text-sm text-white/70 md:text-base">
                  AI-powered voice improv game show
                </p>
              </div>

              {/* Name input */}
              <div className="mb-6">
                <label
                  htmlFor="player-name"
                  className="mb-2 block text-sm font-medium text-white/90"
                >
                  Stage Name
                </label>
                <input
                  id="player-name"
                  type="text"
                  value={playerName}
                  onChange={(e) => setPlayerName(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter' && !isConnecting) {
                      handleStart();
                    }
                  }}
                  placeholder="Enter your contestant name"
                  className="w-full rounded-lg border border-white/20 bg-white/10 px-4 py-3 text-white placeholder:text-white/50 backdrop-blur-sm focus:border-orange-400/50 focus:outline-none focus:ring-2 focus:ring-orange-400/20 transition-all" // Focus border/ring changed to orange
                  disabled={isConnecting}
                />
                <p className="mt-2 text-xs text-white/60">
                  This is how the host will call you on stage.
                </p>
              </div>

              {/* Start button */}
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleStart}
                disabled={isConnecting || !playerName.trim()}
                className="w-full rounded-lg bg-gradient-to-r from-orange-600 to-amber-600 px-6 py-4 font-semibold text-white shadow-lg transition-all hover:from-orange-500 hover:to-amber-500 disabled:cursor-not-allowed disabled:opacity-50 disabled:hover:scale-100" // Orange to Amber gradient
              >
                {isConnecting ? 'Connecting...' : 'Start Improv Battle'}
              </motion.button>
            </div>
          </div>

          {/* Footer attribution */}
          <p className="mt-8 text-center text-xs text-white/40">
            Powered by Murf Falcon & LiveKit
          </p>
        </motion.div>
      </div>
    </div>
  );
};