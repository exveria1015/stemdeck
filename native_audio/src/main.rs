use std::cell::RefCell;
use std::collections::HashMap;
use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::fs::File;
use std::io::BufReader;
use std::path::{Path, PathBuf};
use std::rc::Rc;

use serde::Deserialize;

const RENDER_SAMPLE_RATE: u32 = 48_000;
const RENDER_CHANNELS: usize = 12;
const RENDER_LIMIT: f64 = 0.98;

enum SampleIter {
    Float(hound::WavIntoSamples<BufReader<File>, f32>),
    Int(hound::WavIntoSamples<BufReader<File>, i32>),
}

struct WavSource {
    path: PathBuf,
    channels: usize,
    sample_rate: u32,
    bits_per_sample: u16,
    samples: SampleIter,
    done: bool,
}

impl WavSource {
    fn open(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(format!("{} has no audio channels", path.display()).into());
        }
        let samples = match spec.sample_format {
            hound::SampleFormat::Float => SampleIter::Float(reader.into_samples::<f32>()),
            hound::SampleFormat::Int => SampleIter::Int(reader.into_samples::<i32>()),
        };
        Ok(Self {
            path,
            channels: usize::from(spec.channels),
            sample_rate: spec.sample_rate,
            bits_per_sample: spec.bits_per_sample,
            samples,
            done: false,
        })
    }

    fn next_sample(&mut self) -> Result<Option<f64>, Box<dyn Error>> {
        match &mut self.samples {
            SampleIter::Float(iter) => match iter.next() {
                Some(Ok(sample)) => Ok(Some(f64::from(sample))),
                Some(Err(err)) => Err(Box::new(err)),
                None => Ok(None),
            },
            SampleIter::Int(iter) => match iter.next() {
                Some(Ok(sample)) => {
                    let scale = 2.0_f64.powi(i32::from(self.bits_per_sample) - 1);
                    Ok(Some(f64::from(sample) / scale))
                }
                Some(Err(err)) => Err(Box::new(err)),
                None => Ok(None),
            },
        }
    }

    fn next_frame(&mut self, output_channels: usize) -> Result<Option<Vec<f64>>, Box<dyn Error>> {
        if self.done {
            return Ok(None);
        }
        let mut frame = vec![0.0_f64; self.channels];
        for channel in 0..self.channels {
            match self.next_sample()? {
                Some(sample) => frame[channel] = sample,
                None if channel == 0 => {
                    self.done = true;
                    return Ok(None);
                }
                None => {
                    return Err(format!("{} ended mid-frame", self.path.display()).into());
                }
            }
        }
        if self.channels == output_channels {
            return Ok(Some(frame));
        }
        let mut mapped = vec![0.0_f64; output_channels];
        if self.channels == 1 && output_channels >= 2 {
            mapped[0] = frame[0];
            mapped[1] = frame[0];
            return Ok(Some(mapped));
        }
        for (index, sample) in frame.into_iter().enumerate().take(output_channels) {
            mapped[index] = sample;
        }
        Ok(Some(mapped))
    }
}

struct MixArgs {
    output: PathBuf,
    inputs: Vec<PathBuf>,
}

enum Command {
    Mix(MixArgs),
    Render714(PathBuf),
}

fn parse_args() -> Result<Command, Box<dyn Error>> {
    let mut iter = env::args().skip(1);
    let command = iter
        .next()
        .ok_or("missing command: expected 'mix' or 'render714'")?;
    if command == "--help" || command == "-h" {
        print_help();
        std::process::exit(0);
    }

    if command == "mix" {
        let mut output: Option<PathBuf> = None;
        let mut inputs = Vec::new();
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--output" | "-o" => {
                    output = Some(PathBuf::from(
                        iter.next().ok_or("--output requires a path")?,
                    ));
                }
                "--input" | "-i" => {
                    inputs.push(PathBuf::from(iter.next().ok_or("--input requires a path")?));
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        if inputs.is_empty() {
            return Err("mix requires at least one --input".into());
        }
        return Ok(Command::Mix(MixArgs {
            output: output.ok_or("mix requires --output")?,
            inputs,
        }));
    }

    if command == "render714" {
        let mut plan: Option<PathBuf> = None;
        while let Some(arg) = iter.next() {
            match arg.as_str() {
                "--plan" => {
                    plan = Some(PathBuf::from(iter.next().ok_or("--plan requires a path")?));
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }
        return Ok(Command::Render714(plan.ok_or("render714 requires --plan")?));
    }

    Err(format!("unknown command: {command}").into())
}

fn print_help() {
    println!(
        "Usage:\n  stemdeck-native-audio mix --output OUT.wav --input STEM.wav [--input STEM2.wav ...]\n  stemdeck-native-audio render714 --plan PLAN.json"
    );
}

fn write_sample(
    writer: &mut hound::WavWriter<std::io::BufWriter<File>>,
    sample: f64,
) -> Result<(), hound::Error> {
    let clipped = sample.clamp(-1.0, 1.0);
    let scale = if clipped < 0.0 { 32768.0 } else { 32767.0 };
    writer.write_sample((clipped * scale).round() as i16)
}

fn mix_wav(inputs: &[PathBuf], output: &Path) -> Result<(), Box<dyn Error>> {
    let mut sources: Vec<WavSource> = inputs
        .iter()
        .cloned()
        .map(WavSource::open)
        .collect::<Result<_, _>>()?;
    let sample_rate = sources[0].sample_rate;
    for source in &sources {
        if source.sample_rate != sample_rate {
            return Err(format!(
                "sample-rate mismatch: {} is {} Hz, expected {} Hz",
                source.path.display(),
                source.sample_rate,
                sample_rate
            )
            .into());
        }
    }
    let output_channels = sources
        .iter()
        .map(|source| source.channels)
        .max()
        .unwrap_or(1);
    if output_channels > usize::from(u16::MAX) {
        return Err(format!("too many output channels: {output_channels}").into());
    }
    if let Some(parent) = output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let spec = hound::WavSpec {
        channels: output_channels as u16,
        sample_rate,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output, spec)?;
    loop {
        let mut any = false;
        let mut mixed = vec![0.0_f64; output_channels];
        for source in &mut sources {
            if let Some(frame) = source.next_frame(output_channels)? {
                any = true;
                for (index, sample) in frame.into_iter().enumerate() {
                    mixed[index] += sample;
                }
            }
        }
        if !any {
            break;
        }
        for sample in mixed {
            write_sample(&mut writer, sample)?;
        }
    }
    writer.finalize()?;
    Ok(())
}

#[derive(Clone, Copy)]
struct Biquad {
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    z1: f64,
    z2: f64,
}

impl Biquad {
    fn low_pass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * (freq / sample_rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 - cos_w0) / 2.0;
        let b1 = 1.0 - cos_w0;
        let b2 = (1.0 - cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn high_pass(freq: f64, q: f64, sample_rate: f64) -> Self {
        let w0 = 2.0 * PI * (freq / sample_rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = (1.0 + cos_w0) / 2.0;
        let b1 = -(1.0 + cos_w0);
        let b2 = (1.0 + cos_w0) / 2.0;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn all_pass_hz(freq: f64, width_hz: f64, sample_rate: f64) -> Self {
        let q = (freq / width_hz.max(1.0)).max(0.001);
        let w0 = 2.0 * PI * (freq / sample_rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = 1.0 - alpha;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 + alpha;
        let a0 = 1.0 + alpha;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn peaking(freq: f64, width_octaves: f64, gain_db: f64, sample_rate: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * (freq / sample_rate);
        let octave = 2.0_f64.powf(width_octaves.max(0.001));
        let q = octave.sqrt() / (octave - 1.0).max(1.0e-6);
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let b0 = 1.0 + alpha * a;
        let b1 = -2.0 * cos_w0;
        let b2 = 1.0 - alpha * a;
        let a0 = 1.0 + alpha / a;
        let a1 = -2.0 * cos_w0;
        let a2 = 1.0 - alpha / a;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn from_coeffs(b0: f64, b1: f64, b2: f64, a0: f64, a1: f64, a2: f64) -> Self {
        Self {
            b0: b0 / a0,
            b1: b1 / a0,
            b2: b2 / a0,
            a1: a1 / a0,
            a2: a2 / a0,
            z1: 0.0,
            z2: 0.0,
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        let out = self.b0 * sample + self.z1;
        self.z1 = self.b1 * sample - self.a1 * out + self.z2;
        self.z2 = self.b2 * sample - self.a2 * out;
        out
    }
}

#[derive(Deserialize)]
struct EqBandPlan {
    frequency_hz: f64,
    width_octaves: f64,
    gain_db: f64,
}

#[derive(Deserialize)]
struct GainSegmentPlan {
    start_frame: usize,
    end_frame: usize,
    gain: f64,
}

#[derive(Deserialize)]
struct SpaceBedPlan {
    pan: Vec<[f64; 2]>,
    left_delay_ms: f64,
    right_delay_ms: f64,
    wet_gain: f64,
    allpass_mix: f64,
    #[serde(default)]
    automation: Vec<GainSegmentPlan>,
}

#[derive(Clone, Deserialize)]
struct CompressorPlan {
    threshold: f64,
    ratio: f64,
    attack_ms: f64,
    release_ms: f64,
    mix: f64,
    link: String,
}

#[derive(Deserialize)]
struct SidechainSourcePlan {
    path: PathBuf,
    gain: f64,
}

#[derive(Deserialize)]
struct DuckBandPlan {
    band: String,
    sidechain_gain: f64,
    compressor: CompressorPlan,
}

#[derive(Deserialize)]
struct DuckOperationPlan {
    mode: String,
    #[serde(default)]
    sidechain_filter: String,
    #[serde(default)]
    normalizer: f64,
    #[serde(default)]
    sidechains: Vec<SidechainSourcePlan>,
    #[serde(default)]
    compressor: Option<CompressorPlan>,
    #[serde(default)]
    bands: Vec<DuckBandPlan>,
}

#[derive(Deserialize)]
struct StemRenderPlan {
    path: PathBuf,
    gain: f64,
    pan: Vec<[f64; 2]>,
    lfe_amount: f64,
    lfe_cutoff_hz: f64,
    #[serde(default)]
    eq: Vec<EqBandPlan>,
    #[serde(default)]
    low_tighten: Vec<GainSegmentPlan>,
    #[serde(default)]
    mid_decongest: Vec<GainSegmentPlan>,
    #[serde(default)]
    space_bed: Option<SpaceBedPlan>,
    #[serde(default)]
    ducking: Vec<DuckOperationPlan>,
}

#[derive(Deserialize)]
struct Render714Plan {
    output: PathBuf,
    master_gain: f64,
    stems: Vec<StemRenderPlan>,
}

struct StreamingWav {
    source: WavSource,
    current_index: usize,
    current: Option<[f64; 2]>,
    next: Option<[f64; 2]>,
    total_frames: usize,
}

impl StreamingWav {
    fn open(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        let reader = hound::WavReader::open(&path)?;
        let spec = reader.spec();
        if spec.channels == 0 {
            return Err(format!("{} has no audio channels", path.display()).into());
        }
        let total_frames = reader.duration() as usize;
        drop(reader);
        let mut source = WavSource::open(path)?;
        let current = read_stereo_frame(&mut source)?;
        let next = read_stereo_frame(&mut source)?;
        Ok(Self {
            source,
            current_index: 0,
            current,
            next,
            total_frames,
        })
    }

    fn output_frames(&self) -> usize {
        ((self.total_frames as f64) * f64::from(RENDER_SAMPLE_RATE)
            / f64::from(self.source.sample_rate))
        .ceil() as usize
    }

    fn sample_at(&mut self, output_index: usize) -> Result<[f64; 2], Box<dyn Error>> {
        let source_pos = output_index as f64 * f64::from(self.source.sample_rate)
            / f64::from(RENDER_SAMPLE_RATE);
        let base = source_pos.floor() as usize;
        if base >= self.total_frames {
            return Ok([0.0, 0.0]);
        }
        while self.current_index < base {
            self.current = self.next;
            self.next = read_stereo_frame(&mut self.source)?;
            self.current_index += 1;
        }
        let Some(current) = self.current else {
            return Ok([0.0, 0.0]);
        };
        let frac = source_pos - base as f64;
        if frac <= 1.0e-9 {
            return Ok(current);
        }
        let next = self.next.unwrap_or(current);
        Ok([
            current[0] * (1.0 - frac) + next[0] * frac,
            current[1] * (1.0 - frac) + next[1] * frac,
        ])
    }
}

struct CachedStreamingWav {
    source: StreamingWav,
    cached_index: Option<usize>,
    cached_frame: [f64; 2],
}

impl CachedStreamingWav {
    fn open(path: PathBuf) -> Result<Self, Box<dyn Error>> {
        Ok(Self {
            source: StreamingWav::open(path)?,
            cached_index: None,
            cached_frame: [0.0, 0.0],
        })
    }

    fn output_frames(&self) -> usize {
        self.source.output_frames()
    }

    fn sample_at(&mut self, output_index: usize) -> Result<[f64; 2], Box<dyn Error>> {
        if self.cached_index == Some(output_index) {
            return Ok(self.cached_frame);
        }
        let frame = self.source.sample_at(output_index)?;
        self.cached_index = Some(output_index);
        self.cached_frame = frame;
        Ok(frame)
    }
}

type SharedWav = Rc<RefCell<CachedStreamingWav>>;

#[derive(Default)]
struct SourcePool {
    sources: HashMap<PathBuf, SharedWav>,
}

impl SourcePool {
    fn get(&mut self, path: PathBuf) -> Result<SharedWav, Box<dyn Error>> {
        if let Some(source) = self.sources.get(&path) {
            return Ok(Rc::clone(source));
        }
        let source = Rc::new(RefCell::new(CachedStreamingWav::open(path.clone())?));
        self.sources.insert(path, Rc::clone(&source));
        Ok(source)
    }
}

fn read_stereo_frame(source: &mut WavSource) -> Result<Option<[f64; 2]>, Box<dyn Error>> {
    let Some(frame) = source.next_frame(2)? else {
        return Ok(None);
    };
    Ok(Some([frame[0], frame[1]]))
}

#[derive(Clone)]
struct GainSegment {
    start_frame: usize,
    end_frame: usize,
    gain: f64,
}

struct AutomationGain {
    segments: Vec<GainSegment>,
    index: usize,
}

impl AutomationGain {
    fn from_plan(plan: Vec<GainSegmentPlan>) -> Self {
        let mut segments = plan
            .into_iter()
            .filter(|segment| segment.end_frame > segment.start_frame)
            .map(|segment| GainSegment {
                start_frame: segment.start_frame,
                end_frame: segment.end_frame,
                gain: segment.gain,
            })
            .collect::<Vec<_>>();
        segments.sort_by_key(|segment| segment.start_frame);
        Self { segments, index: 0 }
    }

    fn gain_at(&mut self, frame_index: usize) -> f64 {
        while self.index < self.segments.len() && frame_index >= self.segments[self.index].end_frame
        {
            self.index += 1;
        }
        let Some(segment) = self.segments.get(self.index) else {
            return 1.0;
        };
        if frame_index >= segment.start_frame {
            segment.gain
        } else {
            1.0
        }
    }
}

struct LinkwitzRiley4 {
    first: Biquad,
    second: Biquad,
}

impl LinkwitzRiley4 {
    fn low_pass(freq: f64, sample_rate: f64) -> Self {
        let q = 1.0 / 2.0_f64.sqrt();
        Self {
            first: Biquad::low_pass(freq, q, sample_rate),
            second: Biquad::low_pass(freq, q, sample_rate),
        }
    }

    fn high_pass(freq: f64, sample_rate: f64) -> Self {
        let q = 1.0 / 2.0_f64.sqrt();
        Self {
            first: Biquad::high_pass(freq, q, sample_rate),
            second: Biquad::high_pass(freq, q, sample_rate),
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        self.second.process(self.first.process(sample))
    }
}

struct LowTightenAutomation {
    left_low: LinkwitzRiley4,
    left_high: LinkwitzRiley4,
    right_low: LinkwitzRiley4,
    right_high: LinkwitzRiley4,
    gain: AutomationGain,
}

impl LowTightenAutomation {
    fn new(segments: Vec<GainSegmentPlan>, sample_rate: f64) -> Self {
        Self {
            left_low: LinkwitzRiley4::low_pass(180.0, sample_rate),
            left_high: LinkwitzRiley4::high_pass(180.0, sample_rate),
            right_low: LinkwitzRiley4::low_pass(180.0, sample_rate),
            right_high: LinkwitzRiley4::high_pass(180.0, sample_rate),
            gain: AutomationGain::from_plan(segments),
        }
    }

    fn process(&mut self, frame_index: usize, left: f64, right: f64) -> [f64; 2] {
        let gain = self.gain.gain_at(frame_index);
        [
            self.left_low.process(left) * gain + self.left_high.process(left),
            self.right_low.process(right) * gain + self.right_high.process(right),
        ]
    }
}

struct MidDecongestAutomation {
    left_low: LinkwitzRiley4,
    left_above_low: LinkwitzRiley4,
    left_mid: LinkwitzRiley4,
    left_high: LinkwitzRiley4,
    right_low: LinkwitzRiley4,
    right_above_low: LinkwitzRiley4,
    right_mid: LinkwitzRiley4,
    right_high: LinkwitzRiley4,
    gain: AutomationGain,
}

impl MidDecongestAutomation {
    fn new(segments: Vec<GainSegmentPlan>, sample_rate: f64) -> Self {
        Self {
            left_low: LinkwitzRiley4::low_pass(700.0, sample_rate),
            left_above_low: LinkwitzRiley4::high_pass(700.0, sample_rate),
            left_mid: LinkwitzRiley4::low_pass(3500.0, sample_rate),
            left_high: LinkwitzRiley4::high_pass(3500.0, sample_rate),
            right_low: LinkwitzRiley4::low_pass(700.0, sample_rate),
            right_above_low: LinkwitzRiley4::high_pass(700.0, sample_rate),
            right_mid: LinkwitzRiley4::low_pass(3500.0, sample_rate),
            right_high: LinkwitzRiley4::high_pass(3500.0, sample_rate),
            gain: AutomationGain::from_plan(segments),
        }
    }

    fn process_channel(
        sample: f64,
        low: &mut LinkwitzRiley4,
        above_low: &mut LinkwitzRiley4,
        mid: &mut LinkwitzRiley4,
        high: &mut LinkwitzRiley4,
        gain: f64,
    ) -> f64 {
        let low_part = low.process(sample);
        let above_low_part = above_low.process(sample);
        let mid_part = mid.process(above_low_part);
        let high_part = high.process(above_low_part);
        low_part + mid_part * gain + high_part
    }

    fn process(&mut self, frame_index: usize, left: f64, right: f64) -> [f64; 2] {
        let gain = self.gain.gain_at(frame_index);
        [
            Self::process_channel(
                left,
                &mut self.left_low,
                &mut self.left_above_low,
                &mut self.left_mid,
                &mut self.left_high,
                gain,
            ),
            Self::process_channel(
                right,
                &mut self.right_low,
                &mut self.right_above_low,
                &mut self.right_mid,
                &mut self.right_high,
                gain,
            ),
        ]
    }
}

struct DelayLine {
    samples: Vec<f64>,
    index: usize,
}

impl DelayLine {
    fn new(delay_frames: usize) -> Self {
        Self {
            samples: vec![0.0; delay_frames],
            index: 0,
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        if self.samples.is_empty() {
            return sample;
        }
        let delayed = self.samples[self.index];
        self.samples[self.index] = sample;
        self.index = (self.index + 1) % self.samples.len();
        delayed
    }
}

struct MixedAllPass {
    filter: Biquad,
    mix: f64,
}

impl MixedAllPass {
    fn new(freq: f64, width_hz: f64, mix: f64, sample_rate: f64) -> Self {
        Self {
            filter: Biquad::all_pass_hz(freq, width_hz, sample_rate),
            mix: mix.clamp(0.0, 1.0),
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        sample * (1.0 - self.mix) + self.filter.process(sample) * self.mix
    }
}

struct SpaceBedRenderer {
    pan: [[f64; 2]; RENDER_CHANNELS],
    left_delay: DelayLine,
    right_delay: DelayLine,
    left_allpass_a: MixedAllPass,
    right_allpass_a: MixedAllPass,
    left_allpass_b: MixedAllPass,
    right_allpass_b: MixedAllPass,
    wet_gain: f64,
    automation: AutomationGain,
}

impl SpaceBedRenderer {
    fn from_plan(plan: SpaceBedPlan, sample_rate: f64) -> Result<Self, Box<dyn Error>> {
        if plan.pan.len() != RENDER_CHANNELS {
            return Err(
                format!("space bed pan must contain {RENDER_CHANNELS} channel rows").into(),
            );
        }
        let mut pan = [[0.0_f64; 2]; RENDER_CHANNELS];
        for (index, row) in plan.pan.into_iter().enumerate() {
            pan[index] = row;
        }
        let left_delay = (plan.left_delay_ms.max(0.0) * sample_rate / 1000.0).round() as usize;
        let right_delay = (plan.right_delay_ms.max(0.0) * sample_rate / 1000.0).round() as usize;
        let allpass_mix = plan.allpass_mix.clamp(0.0, 1.0);
        Ok(Self {
            pan,
            left_delay: DelayLine::new(left_delay),
            right_delay: DelayLine::new(right_delay),
            left_allpass_a: MixedAllPass::new(720.0, 420.0, allpass_mix, sample_rate),
            right_allpass_a: MixedAllPass::new(720.0, 420.0, allpass_mix, sample_rate),
            left_allpass_b: MixedAllPass::new(1650.0, 900.0, allpass_mix * 0.75, sample_rate),
            right_allpass_b: MixedAllPass::new(1650.0, 900.0, allpass_mix * 0.75, sample_rate),
            wet_gain: plan.wet_gain,
            automation: AutomationGain::from_plan(plan.automation),
        })
    }

    fn render_into(
        &mut self,
        frame_index: usize,
        left: f64,
        right: f64,
        mixed: &mut [f64; RENDER_CHANNELS],
    ) {
        let mut space_left = 0.5 * left - 0.5 * right;
        let mut space_right = 0.5 * right - 0.5 * left;
        space_left = self.left_delay.process(space_left);
        space_right = self.right_delay.process(space_right);
        space_left = self.left_allpass_a.process(space_left);
        space_right = self.right_allpass_a.process(space_right);
        space_left = self.left_allpass_b.process(space_left);
        space_right = self.right_allpass_b.process(space_right);
        let gain = self.wet_gain * self.automation.gain_at(frame_index);
        space_left *= gain;
        space_right *= gain;
        for (channel, row) in self.pan.iter().enumerate() {
            mixed[channel] += row[0] * space_left + row[1] * space_right;
        }
    }
}

struct StereoFilterChain {
    left: Vec<LinkwitzRiley4>,
    right: Vec<LinkwitzRiley4>,
}

impl StereoFilterChain {
    fn from_name(name: &str, sample_rate: f64) -> Self {
        match name {
            "low180" => Self {
                left: vec![LinkwitzRiley4::low_pass(180.0, sample_rate)],
                right: vec![LinkwitzRiley4::low_pass(180.0, sample_rate)],
            },
            "mid180_2500" => Self {
                left: vec![
                    LinkwitzRiley4::high_pass(180.0, sample_rate),
                    LinkwitzRiley4::low_pass(2500.0, sample_rate),
                ],
                right: vec![
                    LinkwitzRiley4::high_pass(180.0, sample_rate),
                    LinkwitzRiley4::low_pass(2500.0, sample_rate),
                ],
            },
            "high2500" => Self {
                left: vec![LinkwitzRiley4::high_pass(2500.0, sample_rate)],
                right: vec![LinkwitzRiley4::high_pass(2500.0, sample_rate)],
            },
            "low2500" => Self {
                left: vec![LinkwitzRiley4::low_pass(2500.0, sample_rate)],
                right: vec![LinkwitzRiley4::low_pass(2500.0, sample_rate)],
            },
            "high180" => Self {
                left: vec![LinkwitzRiley4::high_pass(180.0, sample_rate)],
                right: vec![LinkwitzRiley4::high_pass(180.0, sample_rate)],
            },
            _ => Self {
                left: Vec::new(),
                right: Vec::new(),
            },
        }
    }

    fn process(&mut self, frame: [f64; 2]) -> [f64; 2] {
        let mut left = frame[0];
        let mut right = frame[1];
        for filter in &mut self.left {
            left = filter.process(left);
        }
        for filter in &mut self.right {
            right = filter.process(right);
        }
        [left, right]
    }
}

struct SidechainSource {
    source: SharedWav,
    gain: f64,
}

struct SidechainMix {
    sources: Vec<SidechainSource>,
    normalizer: f64,
    filter: StereoFilterChain,
}

impl SidechainMix {
    fn from_plan(
        sources: Vec<SidechainSourcePlan>,
        normalizer: f64,
        filter_name: &str,
        sample_rate: f64,
        source_pool: &mut SourcePool,
    ) -> Result<Self, Box<dyn Error>> {
        let sidechain_sources = sources
            .into_iter()
            .map(|source| {
                Ok(SidechainSource {
                    source: source_pool.get(source.path)?,
                    gain: source.gain,
                })
            })
            .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
        Ok(Self {
            sources: sidechain_sources,
            normalizer: if normalizer.abs() < 1.0e-12 {
                1.0
            } else {
                normalizer
            },
            filter: StereoFilterChain::from_name(filter_name, sample_rate),
        })
    }

    fn sample_at(&mut self, output_index: usize) -> Result<[f64; 2], Box<dyn Error>> {
        let mut mixed = [0.0_f64; 2];
        for source in &mut self.sources {
            let frame = source.source.borrow_mut().sample_at(output_index)?;
            mixed[0] += frame[0] * source.gain;
            mixed[1] += frame[1] * source.gain;
        }
        mixed[0] *= self.normalizer;
        mixed[1] *= self.normalizer;
        Ok(self.filter.process(mixed))
    }
}

struct SidechainCompressor {
    params: CompressorPlan,
    envelope: f64,
    attack_coeff: f64,
    release_coeff: f64,
}

impl SidechainCompressor {
    fn new(params: CompressorPlan, sample_rate: f64) -> Self {
        let attack_sec = (params.attack_ms.max(0.1)) / 1000.0;
        let release_sec = (params.release_ms.max(0.1)) / 1000.0;
        Self {
            params,
            envelope: 0.0,
            attack_coeff: (-1.0 / (attack_sec * sample_rate)).exp(),
            release_coeff: (-1.0 / (release_sec * sample_rate)).exp(),
        }
    }

    fn detector_level(&self, sidechain: [f64; 2]) -> f64 {
        if self.params.link == "maximum" {
            sidechain[0].abs().max(sidechain[1].abs())
        } else {
            ((sidechain[0] * sidechain[0] + sidechain[1] * sidechain[1]) * 0.5).sqrt()
        }
    }

    fn gain_for_level(&self, level: f64) -> f64 {
        let threshold = self.params.threshold.max(1.0e-9);
        if level <= threshold || self.params.ratio <= 1.0 {
            return 1.0;
        }
        let over_db = 20.0 * (level / threshold).log10();
        let reduction_db = over_db * (1.0 - 1.0 / self.params.ratio);
        10.0_f64.powf(-reduction_db / 20.0)
    }

    fn process(&mut self, target: [f64; 2], sidechain: [f64; 2]) -> [f64; 2] {
        let detector = self.detector_level(sidechain);
        let coeff = if detector > self.envelope {
            self.attack_coeff
        } else {
            self.release_coeff
        };
        self.envelope = coeff * self.envelope + (1.0 - coeff) * detector;
        let gain = self.gain_for_level(self.envelope);
        let mix = self.params.mix.clamp(0.0, 1.0);
        let wet_gain = gain * mix + (1.0 - mix);
        [target[0] * wet_gain, target[1] * wet_gain]
    }
}

struct BandSplit3 {
    low_l: LinkwitzRiley4,
    high_l: LinkwitzRiley4,
    mid_l: LinkwitzRiley4,
    top_l: LinkwitzRiley4,
    low_r: LinkwitzRiley4,
    high_r: LinkwitzRiley4,
    mid_r: LinkwitzRiley4,
    top_r: LinkwitzRiley4,
}

impl BandSplit3 {
    fn new(sample_rate: f64) -> Self {
        Self {
            low_l: LinkwitzRiley4::low_pass(180.0, sample_rate),
            high_l: LinkwitzRiley4::high_pass(180.0, sample_rate),
            mid_l: LinkwitzRiley4::low_pass(2500.0, sample_rate),
            top_l: LinkwitzRiley4::high_pass(2500.0, sample_rate),
            low_r: LinkwitzRiley4::low_pass(180.0, sample_rate),
            high_r: LinkwitzRiley4::high_pass(180.0, sample_rate),
            mid_r: LinkwitzRiley4::low_pass(2500.0, sample_rate),
            top_r: LinkwitzRiley4::high_pass(2500.0, sample_rate),
        }
    }

    fn process(&mut self, frame: [f64; 2]) -> [[f64; 2]; 3] {
        let low_l = self.low_l.process(frame[0]);
        let above_l = self.high_l.process(frame[0]);
        let mid_l = self.mid_l.process(above_l);
        let high_l = self.top_l.process(above_l);
        let low_r = self.low_r.process(frame[1]);
        let above_r = self.high_r.process(frame[1]);
        let mid_r = self.mid_r.process(above_r);
        let high_r = self.top_r.process(above_r);
        [[low_l, low_r], [mid_l, mid_r], [high_l, high_r]]
    }
}

struct BandSplit5 {
    sub_l: LinkwitzRiley4,
    above_sub_l: LinkwitzRiley4,
    kick_l: LinkwitzRiley4,
    above_kick_l: LinkwitzRiley4,
    upper_l: LinkwitzRiley4,
    above_upper_l: LinkwitzRiley4,
    mid_l: LinkwitzRiley4,
    high_l: LinkwitzRiley4,
    sub_r: LinkwitzRiley4,
    above_sub_r: LinkwitzRiley4,
    kick_r: LinkwitzRiley4,
    above_kick_r: LinkwitzRiley4,
    upper_r: LinkwitzRiley4,
    above_upper_r: LinkwitzRiley4,
    mid_r: LinkwitzRiley4,
    high_r: LinkwitzRiley4,
}

impl BandSplit5 {
    fn new(sample_rate: f64) -> Self {
        Self {
            sub_l: LinkwitzRiley4::low_pass(60.0, sample_rate),
            above_sub_l: LinkwitzRiley4::high_pass(60.0, sample_rate),
            kick_l: LinkwitzRiley4::low_pass(120.0, sample_rate),
            above_kick_l: LinkwitzRiley4::high_pass(120.0, sample_rate),
            upper_l: LinkwitzRiley4::low_pass(180.0, sample_rate),
            above_upper_l: LinkwitzRiley4::high_pass(180.0, sample_rate),
            mid_l: LinkwitzRiley4::low_pass(2500.0, sample_rate),
            high_l: LinkwitzRiley4::high_pass(2500.0, sample_rate),
            sub_r: LinkwitzRiley4::low_pass(60.0, sample_rate),
            above_sub_r: LinkwitzRiley4::high_pass(60.0, sample_rate),
            kick_r: LinkwitzRiley4::low_pass(120.0, sample_rate),
            above_kick_r: LinkwitzRiley4::high_pass(120.0, sample_rate),
            upper_r: LinkwitzRiley4::low_pass(180.0, sample_rate),
            above_upper_r: LinkwitzRiley4::high_pass(180.0, sample_rate),
            mid_r: LinkwitzRiley4::low_pass(2500.0, sample_rate),
            high_r: LinkwitzRiley4::high_pass(2500.0, sample_rate),
        }
    }

    fn process(&mut self, frame: [f64; 2]) -> [[f64; 2]; 5] {
        let sub_l = self.sub_l.process(frame[0]);
        let above_sub_l = self.above_sub_l.process(frame[0]);
        let kick_l = self.kick_l.process(above_sub_l);
        let above_kick_l = self.above_kick_l.process(above_sub_l);
        let upper_l = self.upper_l.process(above_kick_l);
        let above_upper_l = self.above_upper_l.process(above_kick_l);
        let mid_l = self.mid_l.process(above_upper_l);
        let high_l = self.high_l.process(above_upper_l);

        let sub_r = self.sub_r.process(frame[1]);
        let above_sub_r = self.above_sub_r.process(frame[1]);
        let kick_r = self.kick_r.process(above_sub_r);
        let above_kick_r = self.above_kick_r.process(above_sub_r);
        let upper_r = self.upper_r.process(above_kick_r);
        let above_upper_r = self.above_upper_r.process(above_kick_r);
        let mid_r = self.mid_r.process(above_upper_r);
        let high_r = self.high_r.process(above_upper_r);
        [
            [sub_l, sub_r],
            [kick_l, kick_r],
            [upper_l, upper_r],
            [mid_l, mid_r],
            [high_l, high_r],
        ]
    }
}

fn band3_index(name: &str) -> Option<usize> {
    match name {
        "low" => Some(0),
        "mid" => Some(1),
        "high" => Some(2),
        _ => None,
    }
}

fn band5_index(name: &str) -> Option<usize> {
    match name {
        "sub" => Some(0),
        "kick" => Some(1),
        "upper_bass" => Some(2),
        "mid" => Some(3),
        "high" => Some(4),
        _ => None,
    }
}

struct DuckBandProcessor {
    index: usize,
    sidechain_gain: f64,
    compressor: SidechainCompressor,
}

enum DuckOperation {
    Full {
        sidechain: SidechainMix,
        compressor: SidechainCompressor,
    },
    Bands3 {
        sidechain: SidechainMix,
        target_split: BandSplit3,
        sidechain_split: BandSplit3,
        bands: Vec<DuckBandProcessor>,
    },
    Bands5 {
        sidechain: SidechainMix,
        target_split: BandSplit5,
        sidechain_split: BandSplit5,
        bands: Vec<DuckBandProcessor>,
    },
}

impl DuckOperation {
    fn from_plan(
        plan: DuckOperationPlan,
        sample_rate: f64,
        source_pool: &mut SourcePool,
    ) -> Result<Self, Box<dyn Error>> {
        let sidechain = SidechainMix::from_plan(
            plan.sidechains,
            plan.normalizer,
            &plan.sidechain_filter,
            sample_rate,
            source_pool,
        )?;
        match plan.mode.as_str() {
            "full" => Ok(Self::Full {
                sidechain,
                compressor: SidechainCompressor::new(
                    plan.compressor
                        .ok_or("full ducking operation requires compressor")?,
                    sample_rate,
                ),
            }),
            "bands3" => {
                let bands = plan
                    .bands
                    .into_iter()
                    .map(|band| {
                        Ok(DuckBandProcessor {
                            index: band3_index(&band.band)
                                .ok_or_else(|| format!("unsupported bands3 band: {}", band.band))?,
                            sidechain_gain: band.sidechain_gain,
                            compressor: SidechainCompressor::new(band.compressor, sample_rate),
                        })
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                Ok(Self::Bands3 {
                    sidechain,
                    target_split: BandSplit3::new(sample_rate),
                    sidechain_split: BandSplit3::new(sample_rate),
                    bands,
                })
            }
            "bands5" => {
                let bands = plan
                    .bands
                    .into_iter()
                    .map(|band| {
                        Ok(DuckBandProcessor {
                            index: band5_index(&band.band)
                                .ok_or_else(|| format!("unsupported bands5 band: {}", band.band))?,
                            sidechain_gain: band.sidechain_gain,
                            compressor: SidechainCompressor::new(band.compressor, sample_rate),
                        })
                    })
                    .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
                Ok(Self::Bands5 {
                    sidechain,
                    target_split: BandSplit5::new(sample_rate),
                    sidechain_split: BandSplit5::new(sample_rate),
                    bands,
                })
            }
            _ => Err(format!("unsupported ducking operation mode: {}", plan.mode).into()),
        }
    }

    fn process(
        &mut self,
        frame_index: usize,
        target: [f64; 2],
    ) -> Result<[f64; 2], Box<dyn Error>> {
        match self {
            Self::Full {
                sidechain,
                compressor,
            } => {
                let sidechain_frame = sidechain.sample_at(frame_index)?;
                Ok(compressor.process(target, sidechain_frame))
            }
            Self::Bands3 {
                sidechain,
                target_split,
                sidechain_split,
                bands,
            } => {
                let mut target_bands = target_split.process(target);
                let sidechain_bands = sidechain_split.process(sidechain.sample_at(frame_index)?);
                for band in bands {
                    let sidechain_frame = [
                        sidechain_bands[band.index][0] * band.sidechain_gain,
                        sidechain_bands[band.index][1] * band.sidechain_gain,
                    ];
                    target_bands[band.index] = band
                        .compressor
                        .process(target_bands[band.index], sidechain_frame);
                }
                Ok(target_bands.into_iter().fold([0.0, 0.0], |acc, frame| {
                    [acc[0] + frame[0], acc[1] + frame[1]]
                }))
            }
            Self::Bands5 {
                sidechain,
                target_split,
                sidechain_split,
                bands,
            } => {
                let mut target_bands = target_split.process(target);
                let sidechain_bands = sidechain_split.process(sidechain.sample_at(frame_index)?);
                for band in bands {
                    let sidechain_frame = [
                        sidechain_bands[band.index][0] * band.sidechain_gain,
                        sidechain_bands[band.index][1] * band.sidechain_gain,
                    ];
                    target_bands[band.index] = band
                        .compressor
                        .process(target_bands[band.index], sidechain_frame);
                }
                Ok(target_bands.into_iter().fold([0.0, 0.0], |acc, frame| {
                    [acc[0] + frame[0], acc[1] + frame[1]]
                }))
            }
        }
    }
}

struct StemRenderer {
    source: SharedWav,
    gain: f64,
    pan: [[f64; 2]; RENDER_CHANNELS],
    left_eq: Vec<Biquad>,
    right_eq: Vec<Biquad>,
    lfe_left: Biquad,
    lfe_right: Biquad,
    lfe_amount: f64,
    ducking: Vec<DuckOperation>,
    low_tighten: Option<LowTightenAutomation>,
    mid_decongest: Option<MidDecongestAutomation>,
    space_bed: Option<SpaceBedRenderer>,
}

impl StemRenderer {
    fn from_plan(
        plan: StemRenderPlan,
        source_pool: &mut SourcePool,
    ) -> Result<Self, Box<dyn Error>> {
        if plan.pan.len() != RENDER_CHANNELS {
            return Err(format!("pan must contain {RENDER_CHANNELS} channel rows").into());
        }
        let mut pan = [[0.0_f64; 2]; RENDER_CHANNELS];
        for (index, row) in plan.pan.into_iter().enumerate() {
            pan[index] = row;
        }
        let source = source_pool.get(plan.path)?;
        let sample_rate = f64::from(RENDER_SAMPLE_RATE);
        let mut left_eq = Vec::new();
        let mut right_eq = Vec::new();
        for band in plan.eq {
            if band.gain_db.abs() < 0.05 {
                continue;
            }
            left_eq.push(Biquad::peaking(
                band.frequency_hz,
                band.width_octaves,
                band.gain_db,
                sample_rate,
            ));
            right_eq.push(Biquad::peaking(
                band.frequency_hz,
                band.width_octaves,
                band.gain_db,
                sample_rate,
            ));
        }
        let cutoff = plan.lfe_cutoff_hz.max(20.0).min(sample_rate * 0.45);
        let low_tighten = if plan.low_tighten.is_empty() {
            None
        } else {
            Some(LowTightenAutomation::new(plan.low_tighten, sample_rate))
        };
        let mid_decongest = if plan.mid_decongest.is_empty() {
            None
        } else {
            Some(MidDecongestAutomation::new(plan.mid_decongest, sample_rate))
        };
        let space_bed = plan
            .space_bed
            .map(|space_plan| SpaceBedRenderer::from_plan(space_plan, sample_rate))
            .transpose()?;
        let ducking = plan
            .ducking
            .into_iter()
            .map(|operation| DuckOperation::from_plan(operation, sample_rate, source_pool))
            .collect::<Result<Vec<_>, _>>()?;
        Ok(Self {
            source,
            gain: plan.gain,
            pan,
            left_eq,
            right_eq,
            lfe_left: Biquad::low_pass(cutoff, 1.0 / 2.0_f64.sqrt(), sample_rate),
            lfe_right: Biquad::low_pass(cutoff, 1.0 / 2.0_f64.sqrt(), sample_rate),
            lfe_amount: plan.lfe_amount,
            ducking,
            low_tighten,
            mid_decongest,
            space_bed,
        })
    }

    fn output_frames(&self) -> usize {
        self.source.borrow().output_frames()
    }

    fn render_into(
        &mut self,
        output_index: usize,
        mixed: &mut [f64; RENDER_CHANNELS],
    ) -> Result<(), Box<dyn Error>> {
        let [mut left, mut right] = self.source.borrow_mut().sample_at(output_index)?;
        left *= self.gain;
        right *= self.gain;
        for filter in &mut self.left_eq {
            left = filter.process(left);
        }
        for filter in &mut self.right_eq {
            right = filter.process(right);
        }
        for operation in &mut self.ducking {
            [left, right] = operation.process(output_index, [left, right])?;
        }
        if let Some(automation) = &mut self.low_tighten {
            [left, right] = automation.process(output_index, left, right);
        }
        if let Some(automation) = &mut self.mid_decongest {
            [left, right] = automation.process(output_index, left, right);
        }
        for (channel, row) in self.pan.iter().enumerate() {
            mixed[channel] += row[0] * left + row[1] * right;
        }
        if self.lfe_amount.abs() >= 1.0e-9 {
            mixed[3] += self.lfe_amount * self.lfe_left.process(left)
                + self.lfe_amount * self.lfe_right.process(right);
        }
        if let Some(space_bed) = &mut self.space_bed {
            space_bed.render_into(output_index, left, right, mixed);
        }
        Ok(())
    }
}

fn pcm24_sample(sample: f64) -> i32 {
    let clipped = sample.clamp(-1.0, 1.0);
    let scale = if clipped < 0.0 {
        8_388_608.0
    } else {
        8_388_607.0
    };
    (clipped * scale).round() as i32
}

fn render_714_from_plan(path: &Path) -> Result<(), Box<dyn Error>> {
    let plan: Render714Plan = serde_json::from_slice(&std::fs::read(path)?)?;
    if plan.stems.is_empty() {
        return Err("render714 plan has no stems".into());
    }
    if let Some(parent) = plan.output.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    let mut source_pool = SourcePool::default();
    let mut renderers = plan
        .stems
        .into_iter()
        .map(|stem| StemRenderer::from_plan(stem, &mut source_pool))
        .collect::<Result<Vec<_>, _>>()?;
    let output_frames = renderers
        .iter()
        .map(StemRenderer::output_frames)
        .max()
        .unwrap_or(0);
    let spec = hound::WavSpec {
        channels: RENDER_CHANNELS as u16,
        sample_rate: RENDER_SAMPLE_RATE,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&plan.output, spec)?;
    for frame_index in 0..output_frames {
        let mut mixed = [0.0_f64; RENDER_CHANNELS];
        for renderer in &mut renderers {
            renderer.render_into(frame_index, &mut mixed)?;
        }
        for sample in mixed {
            writer.write_sample(pcm24_sample(
                (sample * plan.master_gain).clamp(-RENDER_LIMIT, RENDER_LIMIT),
            ))?;
        }
    }
    writer.finalize()?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    match parse_args()? {
        Command::Mix(args) => mix_wav(&args.inputs, &args.output),
        Command::Render714(plan) => render_714_from_plan(&plan),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_path(name: &str) -> PathBuf {
        let nonce = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("clock")
            .as_nanos();
        env::temp_dir().join(format!(
            "stemdeck-native-audio-{}-{nonce}-{name}",
            std::process::id()
        ))
    }

    fn write_stereo(path: &Path, frames: &[[i16; 2]]) {
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut writer = hound::WavWriter::create(path, spec).expect("create wav");
        for frame in frames {
            writer.write_sample(frame[0]).expect("left");
            writer.write_sample(frame[1]).expect("right");
        }
        writer.finalize().expect("finalize wav");
    }

    fn read_stereo(path: &Path) -> Vec<[i16; 2]> {
        let reader = hound::WavReader::open(path).expect("open wav");
        let samples = reader
            .into_samples::<i16>()
            .collect::<Result<Vec<_>, _>>()
            .expect("samples");
        samples
            .chunks_exact(2)
            .map(|frame| [frame[0], frame[1]])
            .collect()
    }

    fn stereo_pan_rows() -> [[f64; 2]; RENDER_CHANNELS] {
        let mut rows = [[0.0; 2]; RENDER_CHANNELS];
        rows[0] = [1.0, 0.0];
        rows[1] = [0.0, 1.0];
        rows
    }

    fn silent_pan_rows() -> [[f64; 2]; RENDER_CHANNELS] {
        [[0.0; 2]; RENDER_CHANNELS]
    }

    fn read_first_714_frame(path: &Path) -> Vec<i32> {
        let reader = hound::WavReader::open(path).expect("open rendered wav");
        let spec = reader.spec();
        assert_eq!(spec.channels, RENDER_CHANNELS as u16);
        assert_eq!(spec.sample_rate, RENDER_SAMPLE_RATE);
        reader
            .into_samples::<i32>()
            .take(RENDER_CHANNELS)
            .collect::<Result<Vec<_>, _>>()
            .expect("rendered samples")
    }

    fn read_714_frames(path: &Path) -> Vec<Vec<i32>> {
        let reader = hound::WavReader::open(path).expect("open rendered wav");
        let samples = reader
            .into_samples::<i32>()
            .collect::<Result<Vec<_>, _>>()
            .expect("rendered samples");
        samples
            .chunks_exact(RENDER_CHANNELS)
            .map(|frame| frame.to_vec())
            .collect()
    }

    fn rendered_frame_count(path: &Path) -> u32 {
        hound::WavReader::open(path)
            .expect("open rendered wav")
            .duration()
    }

    #[test]
    fn mixes_stereo_wav_inputs_without_normalizing() {
        let a = temp_path("a.wav");
        let b = temp_path("b.wav");
        let out = temp_path("out.wav");
        write_stereo(&a, &[[1000, -1000], [2000, 4000]]);
        write_stereo(&b, &[[3000, 500], [-7000, 1000]]);

        mix_wav(&[a.clone(), b.clone()], &out).expect("mix");

        assert_eq!(read_stereo(&out), vec![[4000, -500], [-5000, 5000]]);
        let _ = std::fs::remove_file(a);
        let _ = std::fs::remove_file(b);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn renders_static_714_pan_plan() {
        let stem = temp_path("stem.wav");
        let plan = temp_path("plan.json");
        let out = temp_path("bed.wav");
        write_stereo(&stem, &[[4096, -2048], [1024, 2048]]);
        let payload = serde_json::json!({
            "output": out,
            "master_gain": 1.0,
            "stems": [{
                "path": stem,
                "gain": 1.0,
                "pan": stereo_pan_rows(),
                "lfe_amount": 0.0,
                "lfe_cutoff_hz": 80.0,
                "eq": []
            }]
        });
        std::fs::write(&plan, serde_json::to_vec(&payload).expect("json")).expect("write plan");

        render_714_from_plan(&plan).expect("render714");

        let frame = read_first_714_frame(&out);
        assert!(frame[0] > 1_000_000);
        assert!(frame[1] < -500_000);
        assert_eq!(frame[2], 0);
        assert_eq!(frame[11], 0);
        assert_eq!(rendered_frame_count(&out), 2);
        let _ = std::fs::remove_file(stem);
        let _ = std::fs::remove_file(plan);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn renders_space_bed_plan_with_automation_gain() {
        let stem = temp_path("space-stem.wav");
        let plan = temp_path("space-plan.json");
        let out = temp_path("space-bed.wav");
        write_stereo(&stem, &[[8192, 0], [8192, 0]]);
        let mut space_pan = silent_pan_rows();
        space_pan[4] = [1.0, 0.0];
        let payload = serde_json::json!({
            "output": out,
            "master_gain": 1.0,
            "stems": [{
                "path": stem,
                "gain": 1.0,
                "pan": silent_pan_rows(),
                "lfe_amount": 0.0,
                "lfe_cutoff_hz": 80.0,
                "eq": [],
                "space_bed": {
                    "pan": space_pan,
                    "left_delay_ms": 0.0,
                    "right_delay_ms": 0.0,
                    "wet_gain": 1.0,
                    "allpass_mix": 0.0,
                    "automation": [{"start_frame": 0, "end_frame": 4, "gain": 0.5}]
                }
            }]
        });
        std::fs::write(&plan, serde_json::to_vec(&payload).expect("json")).expect("write plan");

        render_714_from_plan(&plan).expect("render714");

        let frame = read_first_714_frame(&out);
        assert_eq!(frame[0], 0);
        assert!(frame[4] > 250_000);
        assert!(frame[4] < 600_000);
        assert_eq!(rendered_frame_count(&out), 2);
        let _ = std::fs::remove_file(stem);
        let _ = std::fs::remove_file(plan);
        let _ = std::fs::remove_file(out);
    }

    #[test]
    fn renders_fullband_sidechain_ducking() {
        let target = temp_path("duck-target.wav");
        let sidechain = temp_path("duck-sidechain.wav");
        let plan = temp_path("duck-plan.json");
        let out = temp_path("duck-bed.wav");
        let target_frames = vec![[10_000, 10_000]; 24];
        let sidechain_frames = vec![[20_000, 20_000]; 24];
        write_stereo(&target, &target_frames);
        write_stereo(&sidechain, &sidechain_frames);
        let payload = serde_json::json!({
            "output": out,
            "master_gain": 1.0,
            "stems": [{
                "path": target,
                "gain": 1.0,
                "pan": stereo_pan_rows(),
                "lfe_amount": 0.0,
                "lfe_cutoff_hz": 80.0,
                "eq": [],
                "ducking": [{
                    "mode": "full",
                    "sidechain_filter": "",
                    "normalizer": 1.0,
                    "sidechains": [{"path": sidechain, "gain": 1.0}],
                    "compressor": {
                        "threshold": 0.01,
                        "ratio": 8.0,
                        "attack_ms": 0.1,
                        "release_ms": 30.0,
                        "mix": 1.0,
                        "link": "average"
                    }
                }]
            }]
        });
        std::fs::write(&plan, serde_json::to_vec(&payload).expect("json")).expect("write plan");

        render_714_from_plan(&plan).expect("render714");

        let frames = read_714_frames(&out);
        assert_eq!(frames.len(), 24);
        assert!(frames[20][0] > 0);
        assert!(frames[20][0] < 1_500_000);
        let _ = std::fs::remove_file(target);
        let _ = std::fs::remove_file(sidechain);
        let _ = std::fs::remove_file(plan);
        let _ = std::fs::remove_file(out);
    }
}
