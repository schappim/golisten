package main

import (
	"bytes"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"os/exec"
	"time"

	"github.com/hajimehoshi/go-mp3"
)

// Every supported engine wants 16 kHz mono audio, so that is the one internal
// representation: []float32 in [-1, 1] at targetSampleRate.
const targetSampleRate = 16000

// Size of the canonical 44-byte RIFF/WAVE header this tool writes.
const wavHeaderSize = 44

// Windowed-sinc resampler geometry. taps is the number of sinc zero crossings
// kept either side of the sample point; phases is how finely the fractional
// sample position is quantised before the kernel is looked up. 1024 phases put
// the position error below 0.05% of a sample, which is inaudible to an acoustic
// model, and lets the whole bank be precomputed once instead of calling sin()
// per output sample.
const (
	resampleTaps   = 16
	resamplePhases = 1024

	// Fraction of the new Nyquist frequency the passband is allowed to reach
	// when downsampling. Ending the passband a little early puts the filter's
	// transition band below Nyquist instead of straddling it, which is what
	// turns a ~40 dB rejection at the edge into ~90 dB. It also matches the
	// rolloff soxr and librosa use, so audio arrives at the acoustic model
	// shaped the way its training data was.
	resampleRolloff = 0.945
)

// Audio holds decoded mono PCM alongside the rate it is sampled at.
type Audio struct {
	Samples []float32
	Rate    int
}

// Duration reports the wall-clock length of the decoded audio.
func (a Audio) Duration() time.Duration {
	if a.Rate <= 0 {
		return 0
	}
	return time.Duration(float64(len(a.Samples)) / float64(a.Rate) * float64(time.Second))
}

// runFFmpeg is wired through a package var so tests can stand in for the real
// binary, and so the ffmpeg fallback can be exercised without one installed.
var runFFmpeg = func(input []byte) ([]byte, error) {
	path, err := exec.LookPath("ffmpeg")
	if err != nil {
		return nil, fmt.Errorf("ffmpeg not found in PATH")
	}
	// ffmpeg has to be able to seek its input: MP4/M4A keep the index at the
	// end of the file, and reading those from a pipe fails with "partial file"
	// while still exiting 0 and writing a header-only WAV. Staging the bytes in
	// a real file avoids that entirely.
	tmp, err := os.CreateTemp("", "golisten-input-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp file: %w", err)
	}
	defer os.Remove(tmp.Name())
	if _, err := tmp.Write(input); err != nil {
		tmp.Close()
		return nil, fmt.Errorf("failed to stage audio for ffmpeg: %w", err)
	}
	if err := tmp.Close(); err != nil {
		return nil, fmt.Errorf("failed to stage audio for ffmpeg: %w", err)
	}

	// The output is forced to the exact shape every engine wants, so no further
	// conversion is needed. Writing it to a pipe is fine — WAV streams.
	cmd := exec.Command(path,
		"-hide_banner", "-loglevel", "error",
		"-i", tmp.Name(),
		"-f", "wav", "-acodec", "pcm_s16le",
		"-ar", fmt.Sprint(targetSampleRate), "-ac", "1",
		"pipe:1",
	)
	var out, errBuf bytes.Buffer
	cmd.Stdout = &out
	cmd.Stderr = &errBuf
	if err := cmd.Run(); err != nil {
		return nil, fmt.Errorf("ffmpeg failed: %w: %s", err, truncate(errBuf.String(), 400))
	}
	// ffmpeg reports some decode failures only through an empty output, so a
	// header with no samples has to be treated as a failure rather than as
	// silence.
	if out.Len() <= wavHeaderSize {
		return nil, fmt.Errorf("ffmpeg decoded no audio: %s", truncate(errBuf.String(), 400))
	}
	return out.Bytes(), nil
}

func truncate(s string, max int) string {
	if len(s) <= max {
		return s
	}
	return s[:max] + "..."
}

// audioFormat is the container detected from the leading magic bytes. This is
// structural sniffing only — it never inspects file names or extensions, which
// lie often enough to matter when reading from a pipe.
type audioFormat string

const (
	formatWAV     audioFormat = "wav"
	formatMP3     audioFormat = "mp3"
	formatFLAC    audioFormat = "flac"
	formatOgg     audioFormat = "ogg"
	formatMP4     audioFormat = "mp4"
	formatUnknown audioFormat = "unknown"
)

// detectFormat identifies the container from its header. mp3 detection accepts
// either an ID3 tag or a raw MPEG frame sync, which covers files that were
// stripped of tags.
func detectFormat(data []byte) audioFormat {
	if len(data) < 12 {
		return formatUnknown
	}
	switch {
	case bytes.Equal(data[0:4], []byte("RIFF")) && bytes.Equal(data[8:12], []byte("WAVE")):
		return formatWAV
	case bytes.Equal(data[0:4], []byte("fLaC")):
		return formatFLAC
	case bytes.Equal(data[0:4], []byte("OggS")):
		return formatOgg
	case bytes.Equal(data[4:8], []byte("ftyp")):
		return formatMP4
	case bytes.Equal(data[0:3], []byte("ID3")):
		return formatMP3
	case data[0] == 0xFF && data[1]&0xE0 == 0xE0:
		return formatMP3
	}
	return formatUnknown
}

// mimeForFormat maps a detected container to the Content-Type the cloud
// providers expect when raw bytes are POSTed.
func mimeForFormat(f audioFormat) string {
	switch f {
	case formatWAV:
		return "audio/wav"
	case formatMP3:
		return "audio/mpeg"
	case formatFLAC:
		return "audio/flac"
	case formatOgg:
		return "audio/ogg"
	case formatMP4:
		return "audio/mp4"
	}
	return "application/octet-stream"
}

// fileExtForFormat gives the upload filename extension. The cloud providers
// sniff the bytes themselves, but OpenAI validates the declared extension
// against its supported list, so it has to be honest.
func fileExtForFormat(f audioFormat) string {
	switch f {
	case formatWAV:
		return "wav"
	case formatMP3:
		return "mp3"
	case formatFLAC:
		return "flac"
	case formatOgg:
		return "ogg"
	case formatMP4:
		return "m4a"
	}
	return "bin"
}

// Decode turns an encoded audio file into mono PCM at 16 kHz. WAV and MP3 are
// decoded in-process so the common cases need no external tooling; anything
// else is handed to ffmpeg when it is available.
func Decode(data []byte) (Audio, error) {
	if len(data) == 0 {
		return Audio{}, fmt.Errorf("audio input is empty")
	}

	format := detectFormat(data)
	var (
		audio Audio
		err   error
	)
	switch format {
	case formatWAV:
		audio, err = decodeWAV(data)
	case formatMP3:
		audio, err = decodeMP3(data)
	default:
		wav, ferr := runFFmpeg(data)
		if ferr != nil {
			return Audio{}, fmt.Errorf("cannot decode %s audio natively and %w "+
				"(install ffmpeg, or convert to WAV/MP3 first)", format, ferr)
		}
		audio, err = decodeWAV(wav)
	}
	if err != nil {
		return Audio{}, err
	}
	if len(audio.Samples) == 0 {
		return Audio{}, fmt.Errorf("decoded audio contains no samples")
	}
	return Resample(audio, targetSampleRate)
}

// decodeMP3 decodes an MPEG stream. go-mp3 always emits 16-bit little-endian
// stereo, so the downmix is unconditional.
func decodeMP3(data []byte) (Audio, error) {
	dec, err := mp3.NewDecoder(bytes.NewReader(data))
	if err != nil {
		return Audio{}, fmt.Errorf("failed to decode MP3: %w", err)
	}
	raw, err := io.ReadAll(dec)
	if err != nil {
		return Audio{}, fmt.Errorf("failed to read MP3 stream: %w", err)
	}
	frames := len(raw) / 4 // 2 channels * 2 bytes
	samples := make([]float32, frames)
	for i := 0; i < frames; i++ {
		l := int16(binary.LittleEndian.Uint16(raw[i*4:]))
		r := int16(binary.LittleEndian.Uint16(raw[i*4+2:]))
		samples[i] = (float32(l) + float32(r)) / 2 / 32768
	}
	return Audio{Samples: samples, Rate: dec.SampleRate()}, nil
}

// wav format tags from the RIFF spec.
const (
	wavFormatPCM        = 1
	wavFormatIEEEFloat  = 3
	wavFormatExtensible = 0xFFFE
)

// decodeWAV parses a RIFF/WAVE file, downmixing to mono. It handles the integer
// PCM widths (8/16/24/32-bit) and IEEE float, and skips any non-audio chunks
// (LIST, fact, bext...) that sit between fmt and data.
func decodeWAV(data []byte) (Audio, error) {
	if len(data) < 12 || !bytes.Equal(data[0:4], []byte("RIFF")) || !bytes.Equal(data[8:12], []byte("WAVE")) {
		return Audio{}, fmt.Errorf("not a RIFF/WAVE file")
	}

	var (
		channels   int
		rate       int
		bits       int
		formatTag  int
		pcm        []byte
		haveFormat bool
	)

	pos := 12
	for pos+8 <= len(data) {
		id := string(data[pos : pos+4])
		size := int(binary.LittleEndian.Uint32(data[pos+4 : pos+8]))
		body := pos + 8
		if size < 0 || body+size > len(data) {
			// Truncated final chunk: take whatever is actually there rather
			// than rejecting a file that is merely missing its tail.
			size = len(data) - body
			if size < 0 {
				break
			}
		}
		switch id {
		case "fmt ":
			if size < 16 {
				return Audio{}, fmt.Errorf("WAV fmt chunk is too short (%d bytes)", size)
			}
			formatTag = int(binary.LittleEndian.Uint16(data[body:]))
			channels = int(binary.LittleEndian.Uint16(data[body+2:]))
			rate = int(binary.LittleEndian.Uint32(data[body+4:]))
			bits = int(binary.LittleEndian.Uint16(data[body+14:]))
			if formatTag == wavFormatExtensible && size >= 40 {
				// The real codec lives in the first two bytes of the subformat GUID.
				formatTag = int(binary.LittleEndian.Uint16(data[body+24:]))
			}
			haveFormat = true
		case "data":
			pcm = data[body : body+size]
		}
		pos = body + size
		if size%2 == 1 { // RIFF chunks are word-aligned.
			pos++
		}
	}

	if !haveFormat {
		return Audio{}, fmt.Errorf("WAV file has no fmt chunk")
	}
	if pcm == nil {
		return Audio{}, fmt.Errorf("WAV file has no data chunk")
	}
	if channels <= 0 {
		return Audio{}, fmt.Errorf("WAV file reports %d channels", channels)
	}
	if rate <= 0 {
		return Audio{}, fmt.Errorf("WAV file reports a sample rate of %d Hz", rate)
	}

	readSample, width, err := wavSampleReader(formatTag, bits)
	if err != nil {
		return Audio{}, err
	}

	frameSize := width * channels
	frames := len(pcm) / frameSize
	samples := make([]float32, frames)
	inv := 1 / float32(channels)
	for i := 0; i < frames; i++ {
		base := i * frameSize
		var sum float32
		for c := 0; c < channels; c++ {
			sum += readSample(pcm[base+c*width:])
		}
		samples[i] = sum * inv
	}
	return Audio{Samples: samples, Rate: rate}, nil
}

// wavSampleReader returns a decoder for one sample of the given format plus its
// width in bytes.
func wavSampleReader(formatTag, bits int) (func([]byte) float32, int, error) {
	switch formatTag {
	case wavFormatPCM:
		switch bits {
		case 8:
			// 8-bit PCM is unsigned with a 128 midpoint, unlike every wider width.
			return func(b []byte) float32 { return (float32(b[0]) - 128) / 128 }, 1, nil
		case 16:
			return func(b []byte) float32 {
				return float32(int16(binary.LittleEndian.Uint16(b))) / 32768
			}, 2, nil
		case 24:
			return func(b []byte) float32 {
				v := int32(b[0]) | int32(b[1])<<8 | int32(b[2])<<16
				if v&0x800000 != 0 {
					v |= ^0xFFFFFF // sign-extend the 24-bit value
				}
				return float32(v) / 8388608
			}, 3, nil
		case 32:
			return func(b []byte) float32 {
				return float32(int32(binary.LittleEndian.Uint32(b))) / 2147483648
			}, 4, nil
		}
	case wavFormatIEEEFloat:
		switch bits {
		case 32:
			return func(b []byte) float32 {
				return math.Float32frombits(binary.LittleEndian.Uint32(b))
			}, 4, nil
		case 64:
			return func(b []byte) float32 {
				return float32(math.Float64frombits(binary.LittleEndian.Uint64(b)))
			}, 8, nil
		}
	}
	return nil, 0, fmt.Errorf("unsupported WAV encoding (format tag %d, %d-bit)", formatTag, bits)
}

// Resample converts audio to the target rate with a Blackman-windowed sinc
// kernel. Downsampling scales the cutoff with the rate ratio so the anti-alias
// filter sits below the new Nyquist frequency — going straight to 16 kHz
// without it folds high frequencies back into the speech band and measurably
// hurts recognition.
func Resample(in Audio, rate int) (Audio, error) {
	if rate <= 0 {
		return Audio{}, fmt.Errorf("invalid target sample rate %d", rate)
	}
	if in.Rate == rate {
		return in, nil
	}
	if in.Rate <= 0 {
		return Audio{}, fmt.Errorf("invalid source sample rate %d", in.Rate)
	}

	ratio := float64(rate) / float64(in.Rate)
	cutoff := math.Min(1, ratio)
	if ratio < 1 {
		// Only downsampling needs the guard band; when upsampling there is
		// nothing above the source Nyquist to fold back in.
		cutoff *= resampleRolloff
	}
	// Kernel half-width measured in source samples: as the cutoff drops the
	// filter has to get proportionally longer to keep the same transition band.
	half := int(math.Ceil(float64(resampleTaps) / cutoff))
	bank := buildSincBank(half, cutoff)

	outLen := int(float64(len(in.Samples)) * ratio)
	if outLen <= 0 {
		return Audio{}, fmt.Errorf("audio is too short to resample from %d Hz to %d Hz", in.Rate, rate)
	}
	out := make([]float32, outLen)
	src := in.Samples
	step := float64(in.Rate) / float64(rate)
	width := 2 * half

	for n := range out {
		pos := float64(n) * step
		i0 := int(pos)
		frac := pos - float64(i0)
		phase := int(frac * resamplePhases)
		if phase >= resamplePhases {
			phase = resamplePhases - 1
		}
		kernel := bank[phase*width : (phase+1)*width]

		var sum float32
		// Taps run from i0-half+1 to i0+half; kernel[k] already encodes the
		// fractional offset for this phase.
		start := i0 - half + 1
		for k := 0; k < width; k++ {
			idx := start + k
			if idx < 0 || idx >= len(src) {
				continue
			}
			sum += src[idx] * kernel[k]
		}
		out[n] = sum
	}
	return Audio{Samples: out, Rate: rate}, nil
}

// buildSincBank precomputes the filter taps for every quantised fractional
// position. Layout is phase-major: bank[phase*width + k].
func buildSincBank(half int, cutoff float64) []float32 {
	width := 2 * half
	bank := make([]float32, resamplePhases*width)
	for p := 0; p < resamplePhases; p++ {
		frac := float64(p) / resamplePhases
		var sum float64
		row := bank[p*width : (p+1)*width]
		for k := 0; k < width; k++ {
			// Distance from the output position to this tap, in source samples.
			t := float64(k-half+1) - frac
			v := cutoff * sinc(cutoff*t) * blackman(t, float64(half))
			row[k] = float32(v)
			sum += v
		}
		// Normalise so a DC input passes through at unity gain regardless of
		// where the phase landed; without this the output ripples at the
		// phase-cycle rate.
		if sum != 0 {
			for k := range row {
				row[k] = float32(float64(row[k]) / sum)
			}
		}
	}
	return bank
}

func sinc(x float64) float64 {
	if x == 0 {
		return 1
	}
	px := math.Pi * x
	return math.Sin(px) / px
}

// blackman evaluates the Blackman window at t over a half-width of half,
// returning 0 outside the window.
func blackman(t, half float64) float64 {
	if t < -half || t > half {
		return 0
	}
	// Map [-half, half] onto [0, 1].
	x := (t + half) / (2 * half)
	return 0.42 - 0.5*math.Cos(2*math.Pi*x) + 0.08*math.Cos(4*math.Pi*x)
}

// EncodeWAV serialises mono float samples as 16-bit PCM in a RIFF container —
// the one format every local engine reads, from a 2023 whisper.cpp through to
// transcribe.cpp.
func EncodeWAV(a Audio) []byte {
	const bitsPerSample = 16
	const channels = 1
	dataLen := len(a.Samples) * 2
	buf := bytes.NewBuffer(make([]byte, 0, 44+dataLen))

	byteRate := a.Rate * channels * bitsPerSample / 8
	blockAlign := channels * bitsPerSample / 8

	buf.WriteString("RIFF")
	binary.Write(buf, binary.LittleEndian, uint32(36+dataLen))
	buf.WriteString("WAVE")
	buf.WriteString("fmt ")
	binary.Write(buf, binary.LittleEndian, uint32(16))
	binary.Write(buf, binary.LittleEndian, uint16(wavFormatPCM))
	binary.Write(buf, binary.LittleEndian, uint16(channels))
	binary.Write(buf, binary.LittleEndian, uint32(a.Rate))
	binary.Write(buf, binary.LittleEndian, uint32(byteRate))
	binary.Write(buf, binary.LittleEndian, uint16(blockAlign))
	binary.Write(buf, binary.LittleEndian, uint16(bitsPerSample))
	buf.WriteString("data")
	binary.Write(buf, binary.LittleEndian, uint32(dataLen))

	scratch := make([]byte, 2)
	for _, s := range a.Samples {
		binary.LittleEndian.PutUint16(scratch, uint16(clampToInt16(s)))
		buf.Write(scratch)
	}
	return buf.Bytes()
}

// clampToInt16 scales a float sample into int16 range, saturating rather than
// wrapping so a hot input clips instead of inverting. The scale factor mirrors
// the 32768 used when decoding, which makes an int16 -> float -> int16 round
// trip exact; scaling by 32767 here instead would shift every sample by up to
// an LSB, and that is enough to change what an acoustic model hears.
func clampToInt16(v float32) int16 {
	x := float64(v) * 32768
	if x > 32767 {
		return 32767
	}
	if x < -32768 {
		return -32768
	}
	return int16(math.Round(x))
}

// AudioChunk is a slice of a longer recording, tagged with where it starts on
// the original timeline so its transcript can be shifted back into place.
type AudioChunk struct {
	Audio  Audio
	Offset time.Duration
}

// ChunkAudio splits audio into pieces of at most maxSamples, preferring to cut
// at the quietest point in the second half of each window so a boundary lands
// in a pause rather than mid-word. This is the audio-side mirror of splitting
// long text on sentence boundaries: same idea, measured in signal energy
// instead of punctuation.
func ChunkAudio(a Audio, maxSamples int) []AudioChunk {
	if maxSamples <= 0 || len(a.Samples) <= maxSamples {
		if len(a.Samples) == 0 {
			return nil
		}
		return []AudioChunk{{Audio: a}}
	}

	var chunks []AudioChunk
	// A 20 ms window is long enough to average out a single glottal pulse and
	// short enough to find a real gap between words.
	win := a.Rate / 50
	if win < 1 {
		win = 1
	}

	start := 0
	for len(a.Samples)-start > maxSamples {
		end := start + maxSamples
		split := quietestSplit(a.Samples, start+maxSamples/2, end, win)
		chunks = append(chunks, AudioChunk{
			Audio:  Audio{Samples: a.Samples[start:split], Rate: a.Rate},
			Offset: samplesToDuration(start, a.Rate),
		})
		start = split
	}
	if start < len(a.Samples) {
		chunks = append(chunks, AudioChunk{
			Audio:  Audio{Samples: a.Samples[start:], Rate: a.Rate},
			Offset: samplesToDuration(start, a.Rate),
		})
	}
	return chunks
}

// quietSplitTolerance is how much louder than the quietest candidate a window
// may be and still be treated as an equally good place to cut.
const quietSplitTolerance = 1.5

// quietestSplit picks a cut point in [lo, hi). It finds the quietest window,
// then takes the *latest* window that is roughly as quiet — landing in a pause
// without giving up chunk capacity. Preferring the earliest minimum instead
// would halve every chunk of uniformly quiet audio, doubling the number of
// requests for no benefit. hi is a hard bound because the caller has already
// sized it to the provider's limit.
func quietestSplit(samples []float32, lo, hi, win int) int {
	if lo < 0 {
		lo = 0
	}
	if hi > len(samples) {
		hi = len(samples)
	}
	if lo >= hi || lo+win > hi {
		return hi
	}

	// Half a window of stride keeps the scan cheap while still landing close to
	// the true minimum.
	stride := win / 2
	if stride < 1 {
		stride = 1
	}

	energies := make([]float64, 0, (hi-lo)/stride+1)
	starts := make([]int, 0, cap(energies))
	minEnergy := math.Inf(1)
	for i := lo; i+win <= hi; i += stride {
		var e float64
		for _, s := range samples[i : i+win] {
			e += float64(s) * float64(s)
		}
		energies = append(energies, e)
		starts = append(starts, i)
		minEnergy = math.Min(minEnergy, e)
	}
	if len(energies) == 0 {
		return hi
	}

	// The epsilon lets digital silence, where every window scores exactly zero,
	// match the whole range instead of only the first window.
	threshold := minEnergy*quietSplitTolerance + 1e-12
	for k := len(energies) - 1; k >= 0; k-- {
		if energies[k] <= threshold {
			// Cut in the middle of the quiet window, not at its edge.
			split := starts[k] + win/2
			if split > lo && split < hi {
				return split
			}
			return hi
		}
	}
	return hi
}

func samplesToDuration(n, rate int) time.Duration {
	if rate <= 0 {
		return 0
	}
	return time.Duration(float64(n) / float64(rate) * float64(time.Second))
}
