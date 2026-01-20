import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { cn } from "@/lib/utils"

const PRESET_VOICES = [
    "alba", "marius", "javert", "jean",
    "fantine", "cosette", "eponine", "azelma"
]

interface VoiceSelectorProps {
    selectedVoice: string | null
    customVoice: string
    onVoiceSelect: (voice: string | null) => void
    onCustomVoiceChange: (url: string) => void
}

export function VoiceSelector({
    selectedVoice,
    customVoice,
    onVoiceSelect,
    onCustomVoiceChange
}: VoiceSelectorProps) {
    return (
        <div className="space-y-4">
            <Label className="text-muted-foreground text-xs uppercase tracking-wider font-semibold">
                Voice Selection
            </Label>
            <div className="grid grid-cols-4 gap-2">
                {PRESET_VOICES.map((voice) => (
                    <Button
                        key={voice}
                        variant={selectedVoice === voice ? "default" : "outline"}
                        size="sm"
                        className={cn(
                            "capitalize transition-all duration-200",
                            selectedVoice === voice && "shadow-md shadow-primary/20 scale-[1.02]"
                        )}
                        onClick={() => onVoiceSelect(voice)}
                    >
                        {voice}
                    </Button>
                ))}
            </div>
            <div className="space-y-2">
                <Label htmlFor="custom-voice" className="text-xs text-muted-foreground">
                    Or use a custom URL / Path
                </Label>
                <Input
                    id="custom-voice"
                    placeholder="hf://kyutai/tts-voices/voice.wav"
                    value={customVoice}
                    onChange={(e) => onCustomVoiceChange(e.target.value)}
                    className="bg-muted/30 border-muted-foreground/20 focus:border-primary/50"
                />
            </div>
        </div>
    )
}
