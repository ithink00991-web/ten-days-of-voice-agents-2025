'use client';

import { useEffect, useRef, useState } from 'react';
import { motion } from 'framer-motion'; // Using framer-motion to avoid errors
import { useRoomContext, useRemoteParticipants } from '@livekit/components-react';
import {
  MicrophoneIcon,
  MicrophoneSlashIcon,
  PhoneDisconnectIcon,
} from '@phosphor-icons/react/dist/ssr';
import type { AppConfig } from '@/app-config';
import { useChatMessages } from '@/hooks/useChatMessages';
import { useConnectionTimeout } from '@/hooks/useConnectionTimout';
import { useDebugMode } from '@/hooks/useDebug';
import { cn } from '@/lib/utils';

const IN_DEVELOPMENT = process.env.NODE_ENV !== 'production';

interface SessionViewProps {
  appConfig: AppConfig;
  onAnimationComplete?: () => void;
}

export const SessionView = ({ appConfig, onAnimationComplete }: SessionViewProps) => {
  useConnectionTimeout(200_000);
  useDebugMode({ enabled: IN_DEVELOPMENT });

  const room = useRoomContext();
  const messages = useChatMessages();
  const participants = useRemoteParticipants();
  const transcriptRef = useRef<HTMLDivElement>(null);
  const [isMuted, setIsMuted] = useState(false);
  const [gamePhase, setGamePhase] = useState<'intro' | 'awaiting_improv' | 'reacting' | 'done'>('intro');
  const [currentRound, setCurrentRound] = useState(0);
  const [maxRounds] = useState(3);
  const [currentScenario, setCurrentScenario] = useState<string | null>(null);
  const [isHostSpeaking, setIsHostSpeaking] = useState(false);

  // Check if host is speaking (agent participant)
  const agentParticipant = participants.find((p) => p.isAgent);
  useEffect(() => {
    if (agentParticipant && agentParticipant.audioTracks) {
      const audioTracks = Array.from(agentParticipant.audioTracks.values());
      const hasActiveAudio = audioTracks.some(
        (track) => track.track && track.track.mediaStreamTrack && track.track.mediaStreamTrack.readyState === 'live'
      );
      setIsHostSpeaking(hasActiveAudio);
    } else {
      setIsHostSpeaking(false);
    }
  }, [agentParticipant, messages]);

  // Auto-scroll transcript
  useEffect(() => {
    if (transcriptRef.current) {
      transcriptRef.current.scrollTop = transcriptRef.current.scrollHeight;
    }
  }, [messages]);

  // Parse game state from messages (simplified - in production, use data messages)
  useEffect(() => {
    const lastMessage = messages[messages.length - 1];
    if (lastMessage && !lastMessage.from?.isLocal) {
      const text = lastMessage.message.toLowerCase();
      // Extract round number more reliably
      if (text.includes('round') && text.includes('of')) {
        // Try multiple patterns to catch "Round 1 of 3", "round 2", etc.
        const roundMatch = text.match(/round\s+(\d+)\s+of\s+(\d+)/i) || text.match(/round\s+(\d+)/i);
        if (roundMatch) {
          const roundNum = parseInt(roundMatch[1]);
          if (roundNum >= 1 && roundNum <= 3) {
            setCurrentRound(roundNum);
          }
        }
      }
      if (text.includes('scenario') || text.includes('you are')) {
        setCurrentScenario(lastMessage.message);
        setGamePhase('awaiting_improv');
      }
      if (text.includes('that was') || text.includes('great') || text.includes('interesting') || text.includes('unique')) {
        setGamePhase('reacting');
      }
      // Detect closing summary
      if (text.includes('thank you for playing') || text.includes('end of our three rounds') || text.includes('closing summary')) {
        setGamePhase('done');
      }
    }
  }, [messages]);

  const toggleMute = async () => {
    const enabled = !isMuted;
    await room.localParticipant.setMicrophoneEnabled(enabled);
    setIsMuted(!enabled);
  };

  const handleLeave = () => {
    room.disconnect();
    if (onAnimationComplete) {
      onAnimationComplete();
    }
  };

  const getPhaseLabel = () => {
    switch (gamePhase) {
      case 'intro':
        return 'Intro';
      case 'awaiting_improv':
        return 'Playing scene';
      case 'reacting':
        return 'Host reacting';
      case 'done':
        return 'Summary';
      default:
        return 'In game';
    }
  };

  return (
    <div className="relative h-screen w-full overflow-hidden bg-gradient-to-br from-gray-900 via-black to-gray-900"> {/* Black/Gray theme */}
      {/* Background effects */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_30%_20%,rgba(249,115,22,0.2),transparent_50%)]" /> {/* Orange accent */}
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_70%_80%,rgba(251,191,36,0.15),transparent_50%)]" /> {/* Amber accent */}

      <div className="relative z-10 flex h-full flex-col">
        {/* Stage Panel */}
        <div className="border-b border-white/10 bg-white/5 backdrop-blur-sm">
          <div className="mx-auto max-w-7xl px-6 py-6">
            <div className="flex items-center justify-between">
              <div className="flex-1">
                <div className="mb-2 flex items-center gap-3">
                  <span className="inline-flex items-center rounded-full bg-orange-500/20 px-3 py-1 text-xs font-semibold text-orange-300"> {/* Orange change */}
                    LIVE
                  </span>
                  <span className="text-sm font-medium text-white/60">{getPhaseLabel()}</span>
                </div>
                <h2 className="mb-1 text-2xl font-bold text-white">
                  {currentRound > 0 ? `Round ${currentRound} of ${maxRounds}` : 'Improv Battle'}
                </h2>
                {currentScenario && (
                  <p className="text-sm text-white/80 line-clamp-2">{currentScenario}</p>
                )}
              </div>

              {/* Host speaking indicator */}
              <div className="flex items-center gap-2">
                {isHostSpeaking && (
                  <motion.div
                    animate={{ scale: [1, 1.2, 1] }}
                    transition={{ repeat: Infinity, duration: 1.5 }}
                    className="flex items-center gap-2 rounded-full bg-orange-500/20 px-4 py-2" // Orange change
                  >
                    <div className="h-2 w-2 rounded-full bg-orange-400" /> {/* Orange change */}
                    <span className="text-xs font-medium text-orange-300">Host speaking</span> {/* Orange change */}
                  </motion.div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Main Content Area */}
        <div className="flex flex-1 overflow-hidden">
          {/* Transcript Panel */}
          <div className="flex-1 overflow-y-auto px-6 py-6">
            <div ref={transcriptRef} className="mx-auto max-w-3xl space-y-4">
              {messages.length === 0 ? (
                <div className="flex h-full items-center justify-center">
                  <p className="text-white/40">Waiting for the show to begin...</p>
                </div>
              ) : (
                messages.map((message) => {
                  const isLocal = message.from?.isLocal ?? false;
                  const isHost = !isLocal && message.from?.isAgent;

                  return (
                    <motion.div
                      key={message.id}
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      className={cn(
                        'flex',
                        isLocal ? 'justify-end' : 'justify-start'
                      )}
                    >
                      <div
                        className={cn(
                          'max-w-[80%] rounded-2xl px-4 py-3',
                          isHost
                            ? 'bg-orange-500/20 text-white' // Host/Agent bubble in Orange
                            : isLocal
                            ? 'bg-amber-500/20 text-white' // Local Player bubble in Amber/Gold
                            : 'bg-white/10 text-white/90'
                        )}
                      >
                        <div className="mb-1 text-xs font-semibold opacity-70">
                          {isHost ? 'Host' : isLocal ? 'You' : 'Player'}
                        </div>
                        <div className="text-sm leading-relaxed">{message.message}</div>
                      </div>
                    </motion.div>
                  );
                })
              )}
            </div>
          </div>
        </div>

        {/* Custom Control Bar */}
        <div className="border-t border-white/10 bg-white/5 backdrop-blur-sm">
          <div className="mx-auto max-w-7xl px-6 py-4">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-3">
                {/* Status Chip */}
                <div className="rounded-full bg-white/10 px-4 py-2">
                  <span className="text-xs font-medium text-white/80">{getPhaseLabel()}</span>
                </div>
              </div>

              <div className="flex items-center gap-3">
                {/* Mic Toggle */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={toggleMute}
                  className={cn(
                    'flex h-12 w-12 items-center justify-center rounded-full transition-all',
                    isMuted
                      ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30' // Muted remains Red for safety/warning
                      : 'bg-white/10 text-white hover:bg-white/20' // Active remains white/subtle
                  )}
                >
                  {isMuted ? (
                    <MicrophoneSlashIcon size={20} weight="fill" />
                  ) : (
                    <MicrophoneIcon size={20} weight="fill" />
                  )}
                </motion.button>

                {/* Leave Button */}
                <motion.button
                  whileHover={{ scale: 1.05 }}
                  whileTap={{ scale: 0.95 }}
                  onClick={handleLeave}
                  className="flex h-12 items-center gap-2 rounded-full bg-red-500/20 px-6 text-red-400 transition-all hover:bg-red-500/30" // Leave remains Red for warning
                >
                  <PhoneDisconnectIcon size={20} weight="fill" />
                  <span className="text-sm font-medium">Leave</span>
                </motion.button>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};