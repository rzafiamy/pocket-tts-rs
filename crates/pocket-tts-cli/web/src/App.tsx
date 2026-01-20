import { useState } from "react"
import { useTTSStream } from "@/hooks/use-tts-stream"
import { VoiceSelector } from "@/components/tts/voice-selector"
import { BufferVisualizer } from "@/components/tts/buffer-visualizer"
import { Card, CardContent, CardDescription, CardHeader, CardTitle, CardFooter } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Alert, AlertDescription, AlertTitle } from "@/components/ui/alert"
import {
    PlayIcon,
    SquareIcon,
    DownloadIcon,
    Volume2Icon,
    AlertCircleIcon,
    GithubIcon,
    MessageSquare
} from "lucide-react"
import { Badge } from "@/components/ui/badge"

export default function App() {
    const [text, setText] = useState("Hello world! I am Pocket TTS running in Rust. I'm blazingly fast on CPU.")
    const [selectedVoice, setSelectedVoice] = useState<string | null>("alba")
    const [customVoice, setCustomVoice] = useState("")

    const {
        state,
        error,
        bufferSize,
        generationTime,
        generate,
        stop,
        downloadWav,
        hasAudio
    } = useTTSStream()

    const handleGenerate = () => {
        const voice = customVoice || selectedVoice || "alba"
        generate(text, voice)
    }

    const handleStop = () => {
        stop()
    }

    const handleVoiceSelect = (voice: string | null) => {
        setSelectedVoice(voice)
        if (voice) setCustomVoice("")
    }

    const handleCustomVoiceChange = (url: string) => {
        setCustomVoice(url)
        if (url) setSelectedVoice(null)
    }

    const isIdle = state === 'idle' || state === 'finished' || state === 'error'

    return (
        <div className="min-h-screen bg-background selection:bg-primary/20 flex flex-col items-center justify-center p-4 md:p-8 font-sans transition-colors duration-500">
            {/* Background Gradient */}
            <div className="fixed inset-0 overflow-hidden -z-10 bg-[radial-gradient(circle_at_top_left,var(--color-primary)_0%,transparent_30%),radial-gradient(circle_at_bottom_right,oklch(0.5_0.1_260)_0%,transparent_30%)] opacity-[0.05]" />

            <main className="w-full max-w-2xl animate-in fade-in slide-in-from-bottom-4 duration-1000">
                <div className="flex items-center justify-between mb-8">
                    <div className="flex items-center gap-3">
                        <div className="p-2.5 bg-primary/10 rounded-xl border border-primary/20 shadow-inner">
                            <Volume2Icon className="w-6 h-6 text-primary" />
                        </div>
                        <div>
                            <h1 className="text-2xl font-bold tracking-tight text-foreground/90 flex items-center gap-2">
                                Pocket TTS
                                <Badge variant="outline" className="text-[10px] py-0 font-medium border-primary/20 text-primary/70">CANDLE PORT</Badge>
                            </h1>
                            <p className="text-xs text-muted-foreground font-medium">Blazingly fast CPU Text-to-Speech</p>
                        </div>
                    </div>
                    <div className="flex items-center gap-2">
                        <a
                            href="https://github.com/babybirdprd/pocket-tts-candle"
                            target="_blank"
                            rel="noreferrer"
                            className="inline-flex items-center justify-center size-9 rounded-full hover:bg-muted/50 transition-colors"
                        >
                            <GithubIcon className="w-4 h-4" />
                        </a>
                    </div>
                </div>

                <div className="grid gap-6">
                    <Card className="border-muted-foreground/10 bg-card/50 backdrop-blur-xl shadow-2xl shadow-primary/5 ring-1 ring-white/10 overflow-hidden">
                        <CardHeader className="pb-4">
                            <div className="flex items-center justify-between">
                                <div className="space-y-1">
                                    <CardTitle className="text-lg flex items-center gap-2">
                                        <MessageSquare className="w-4 h-4 text-primary/70" />
                                        Input Text
                                    </CardTitle>
                                    <CardDescription>What should I say?</CardDescription>
                                </div>
                            </div>
                        </CardHeader>
                        <CardContent className="space-y-6">
                            <div className="relative group">
                                <Textarea
                                    placeholder="Type something amazing..."
                                    className="min-h-[140px] text-base leading-relaxed resize-none bg-muted/20 border-muted-foreground/10 group-hover:border-primary/30 transition-all duration-300 focus-visible:ring-primary/20 focus-visible:ring-offset-0 focus-visible:border-primary/50"
                                    value={text}
                                    onChange={(e) => setText(e.target.value)}
                                />
                                <div className="absolute bottom-3 right-3 text-[10px] font-mono text-muted-foreground/50 opacity-0 group-focus-within:opacity-100 transition-opacity">
                                    {text.length} characters
                                </div>
                            </div>

                            <VoiceSelector
                                selectedVoice={selectedVoice}
                                customVoice={customVoice}
                                onVoiceSelect={handleVoiceSelect}
                                onCustomVoiceChange={handleCustomVoiceChange}
                            />

                            <BufferVisualizer
                                state={state}
                                bufferSize={bufferSize}
                                generationTime={generationTime}
                            />

                            {error && (
                                <Alert variant="destructive" className="animate-in fade-in zoom-in-95 duration-300">
                                    <AlertCircleIcon className="h-4 w-4" />
                                    <AlertTitle>Generation Error</AlertTitle>
                                    <AlertDescription>
                                        {error}
                                    </AlertDescription>
                                </Alert>
                            )}
                        </CardContent>
                        <CardFooter className="bg-muted/5 border-t border-muted-foreground/5 py-4 flex flex-col gap-3">
                            <div className="flex gap-2 w-full">
                                {isIdle ? (
                                    <Button
                                        className="flex-1 h-12 text-base font-semibold transition-all duration-300 shadow-lg shadow-primary/25 hover:shadow-primary/40 group active:scale-[0.98]"
                                        onClick={handleGenerate}
                                    >
                                        <PlayIcon className="w-4 h-4 transition-transform group-hover:scale-110" data-icon="inline-start" />
                                        Generate Audio
                                    </Button>
                                ) : (
                                    <Button
                                        variant="destructive"
                                        className="flex-1 h-12 text-base font-semibold group active:scale-[0.98]"
                                        onClick={handleStop}
                                    >
                                        <SquareIcon className="w-4 h-4 group-hover:scale-110" data-icon="inline-start" />
                                        {state === 'buffering' ? 'Cancel Buffering' : 'Stop Playback'}
                                    </Button>
                                )}

                                <Button
                                    variant="outline"
                                    className="h-12 w-12 p-0 border-muted-foreground/10 hover:bg-primary/5 transition-all duration-300 active:scale-[0.98]"
                                    disabled={!hasAudio}
                                    onClick={downloadWav}
                                    title="Download WAV"
                                >
                                    <DownloadIcon className="w-5 h-5 text-muted-foreground group-hover:text-primary transition-colors" />
                                </Button>
                            </div>
                        </CardFooter>
                    </Card>
                </div>

                <footer className="mt-12 text-center space-y-4">
                    <p className="text-[10px] text-muted-foreground uppercase tracking-[0.2em] font-bold">
                        Powered by Candle & PyO3 • Zero Python Runtime
                    </p>
                </footer>
            </main>
        </div>
    )
}