import { useState, useCallback, useRef, useMemo } from 'react';

// The Worklet code as a string to avoid external file loading issues in embedded environments
const WORKLET_CODE = `
class PCMProcessor extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.buffer = [];
    this.hasStarted = false;
    this.isBuffering = false;
    
    // Default 3s initial buffer, can be updated via messages if needed
    this.startThreshold = (options.processorOptions && options.processorOptions.startThreshold) || (24000 * 3.0);
    this.resumeThreshold = 24000 * 0.5;
    
    this.port.onmessage = (e) => {
      if (e.data.type === 'samples') {
        const samples = e.data.samples;
        for (let i = 0; i < samples.length; i++) {
          this.buffer.push(samples[i]);
        }
      } else if (e.data.type === 'config') {
        if (e.data.startThreshold) this.startThreshold = e.data.startThreshold;
      }
    };
  }

  process(inputs, outputs) {
    const output = outputs[0];
    const channel = output[0];

    // Report buffer size periodically (approx every 100ms)
    const frameCount = typeof currentFrame !== 'undefined' ? currentFrame : (this.frameCount || 0);
    this.frameCount = (this.frameCount || 0) + 1;

    if (frameCount % 40 === 0) {
      this.port.postMessage({ type: 'buffer_pga', length: this.buffer.length });
    }

    // 1. Initial buffering state
    if (!this.hasStarted) {
      if (this.buffer.length < this.startThreshold) {
        return true; 
      }
      this.hasStarted = true;
      this.isBuffering = false;
      this.port.postMessage({ type: 'state', state: 'playing' });
    }

    // 2. Re-buffering state
    if (this.isBuffering) {
      if (this.buffer.length < this.resumeThreshold) {
        this.fillSilence(channel);
        return true;
      }
      this.isBuffering = false;
      this.port.postMessage({ type: 'state', state: 'playing' });
    }

    // 3. Playback state
    const samplesToCopy = Math.min(channel.length, this.buffer.length);
    for (let i = 0; i < samplesToCopy; i++) {
      channel[i] = this.buffer[i];
    }

    if (samplesToCopy > 0) {
      this.buffer.splice(0, samplesToCopy);
    }

    if (samplesToCopy < channel.length) {
      for (let i = samplesToCopy; i < channel.length; i++) {
        channel[i] = 0;
      }
      this.isBuffering = true;
      this.port.postMessage({ type: 'state', state: 'buffering' });
    }

    return true;
  }

  fillSilence(channel) {
    for (let i = 0; i < channel.length; i++) {
      channel[i] = 0;
    }
  }
}

registerProcessor('pcm-processor', PCMProcessor);
`;

export type StreamState = 'idle' | 'connecting' | 'buffering' | 'playing' | 'finished' | 'error';

export function useTTSStream() {
  const [state, setState] = useState<StreamState>('idle');
  const [error, setError] = useState<string | null>(null);
  const [bufferSize, setBufferSize] = useState(0);
  const [generationTime, setGenerationTime] = useState(0);

  const audioCtxRef = useRef<AudioContext | null>(null);
  const workletNodeRef = useRef<AudioWorkletNode | null>(null);
  const currentChunksRef = useRef<Uint8Array[]>([]);

  const stop = useCallback(() => {
    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
      workletNodeRef.current = null;
    }
    setState('idle');
  }, []);

  const initAudio = useCallback(async (startThreshold: number) => {
    if (!audioCtxRef.current) {
      const Ctx = (window as any).AudioContext || (window as any).webkitAudioContext;
      audioCtxRef.current = new Ctx({ sampleRate: 24000 });

      const blob = new Blob([WORKLET_CODE], { type: 'application/javascript' });
      const url = URL.createObjectURL(blob);
      try {
        await audioCtxRef.current!.audioWorklet.addModule(url);
      } finally {
        URL.revokeObjectURL(url);
      }
    }

    if (!audioCtxRef.current) return;

    if (audioCtxRef.current.state === 'suspended') {
      await audioCtxRef.current.resume();
    }

    if (workletNodeRef.current) {
      workletNodeRef.current.disconnect();
    }

    workletNodeRef.current = new AudioWorkletNode(audioCtxRef.current, 'pcm-processor', {
      processorOptions: { startThreshold }
    });

    workletNodeRef.current.port.onmessage = (e) => {
      if (e.data.type === 'buffer_pga') {
        setBufferSize(e.data.length);
      } else if (e.data.type === 'state') {
        if (e.data.state === 'buffering') setState('buffering');
        else if (e.data.state === 'playing') setState('playing');
      }
    };

    workletNodeRef.current.connect(audioCtxRef.current.destination);
  }, []);

  const generate = useCallback(async (text: string, voice: string) => {
    setState('connecting');
    setError(null);
    currentChunksRef.current = [];
    const startTime = performance.now();

    try {
      await initAudio(24000 * 3.0);

      const response = await fetch('/stream', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, voice })
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || 'Failed to start stream');
      }

      const reader = response.body?.getReader();
      if (!reader) throw new Error('No readable stream');

      setState('buffering');

      let leftover = new Uint8Array(0);
      let samplesReceived = 0;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const combined = new Uint8Array(leftover.length + value.length);
        combined.set(leftover);
        combined.set(value, leftover.length);

        const validLen = combined.length - (combined.length % 2);
        const chunk = combined.slice(0, validLen);
        leftover = combined.slice(validLen);

        if (chunk.length > 0) {
          currentChunksRef.current.push(chunk);

          const view = new DataView(chunk.buffer);
          const numSamples = chunk.length / 2;
          const floatSamples = new Float32Array(numSamples);

          for (let i = 0; i < numSamples; i++) {
            const int16 = view.getInt16(i * 2, true);
            floatSamples[i] = int16 / 32768.0;
          }

          samplesReceived += numSamples;

          const elapsed = (performance.now() - startTime) / 1000;
          const samplesPerSec = samplesReceived / elapsed;

          if (samplesPerSec < 22000) {
            workletNodeRef.current?.port.postMessage({
              type: 'config',
              startThreshold: 24000 * 5.0
            });
          }

          workletNodeRef.current?.port.postMessage({
            type: 'samples',
            samples: floatSamples
          });
        }
      }

      setGenerationTime((performance.now() - startTime) / 1000);
      setState('finished');

    } catch (err: any) {
      setError(err.message);
      setState('error');
    }
  }, [initAudio]);

  const downloadWav = useCallback(() => {
    if (currentChunksRef.current.length === 0) return;

    const totalLen = currentChunksRef.current.reduce((acc, c) => acc + c.length, 0);
    const wavData = new Uint8Array(44 + totalLen);
    const view = new DataView(wavData.buffer);

    const writeString = (offset: number, s: string) => {
      for (let i = 0; i < s.length; i++) view.setUint8(offset + i, s.charCodeAt(i));
    };

    writeString(0, 'RIFF');
    view.setUint32(4, 36 + totalLen, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, 1, true);
    view.setUint32(24, 24000, true);
    view.setUint32(28, 24000 * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, totalLen, true);

    let offset = 44;
    for (const chunk of currentChunksRef.current) {
      wavData.set(chunk, offset);
      offset += chunk.length;
    }

    const blob = new Blob([wavData], { type: 'audio/wav' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'pocket-tts-output.wav';
    a.click();
    URL.revokeObjectURL(url);
  }, []);

  return useMemo(() => ({
    state,
    error,
    bufferSize,
    generationTime,
    generate,
    stop,
    downloadWav,
    hasAudio: currentChunksRef.current.length > 0
  }), [state, error, bufferSize, generationTime, generate, stop, downloadWav]);
}
