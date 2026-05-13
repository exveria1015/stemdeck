use std::env;
use std::error::Error;
use std::f64::consts::PI;
use std::fs;
use std::io::{Seek, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

const RENDER_TRUE_PEAK_MARGIN_DB: f64 = 0.5;

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
    fn high_shelf(gain_db: f64, q: f64, fc: f64, rate: f64) -> Self {
        let a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * (fc / rate);
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let sqrt_a = a.sqrt();
        let b0 = a * ((a + 1.0) + (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha);
        let b1 = -2.0 * a * ((a - 1.0) + (a + 1.0) * cos_w0);
        let b2 = a * ((a + 1.0) + (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha);
        let a0 = (a + 1.0) - (a - 1.0) * cos_w0 + 2.0 * sqrt_a * alpha;
        let a1 = 2.0 * ((a - 1.0) - (a + 1.0) * cos_w0);
        let a2 = (a + 1.0) - (a - 1.0) * cos_w0 - 2.0 * sqrt_a * alpha;
        Self::from_coeffs(b0, b1, b2, a0, a1, a2)
    }

    fn high_pass(gain_db: f64, q: f64, fc: f64, rate: f64) -> Self {
        let _a = 10.0_f64.powf(gain_db / 40.0);
        let w0 = 2.0 * PI * (fc / rate);
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

#[derive(Clone, Copy)]
struct KWeightChannel {
    shelf: Biquad,
    high_pass: Biquad,
}

impl KWeightChannel {
    fn new(sample_rate: u32) -> Self {
        let rate = sample_rate as f64;
        Self {
            shelf: Biquad::high_shelf(4.0, 1.0 / 2.0_f64.sqrt(), 1500.0, rate),
            high_pass: Biquad::high_pass(0.0, 0.5, 38.0, rate),
        }
    }

    fn process(&mut self, sample: f64) -> f64 {
        self.high_pass.process(self.shelf.process(sample))
    }
}

struct Args {
    bed: PathBuf,
    lfe_fold_gain: f64,
    target_i: f64,
    target_tp: f64,
    target_lra: f64,
    analysis_json: Option<PathBuf>,
    analysis_svg: Option<PathBuf>,
    render_wav: Option<PathBuf>,
}

fn parse_args() -> Result<Args, Box<dyn Error>> {
    let mut bed: Option<PathBuf> = None;
    let mut lfe_fold_gain = 0.45;
    let mut target_i = -14.0;
    let mut target_tp = -1.0;
    let mut target_lra = 11.0;
    let mut analysis_json: Option<PathBuf> = None;
    let mut analysis_svg: Option<PathBuf> = None;
    let mut render_wav: Option<PathBuf> = None;
    let mut iter = env::args().skip(1);
    while let Some(arg) = iter.next() {
        let value = match arg.as_str() {
            "--bed" => {
                bed = Some(PathBuf::from(iter.next().ok_or("--bed requires a path")?));
                continue;
            }
            "--analysis-json" => {
                analysis_json = Some(PathBuf::from(
                    iter.next().ok_or("--analysis-json requires a path")?,
                ));
                continue;
            }
            "--analysis-svg" => {
                analysis_svg = Some(PathBuf::from(
                    iter.next().ok_or("--analysis-svg requires a path")?,
                ));
                continue;
            }
            "--render-wav" => {
                render_wav = Some(PathBuf::from(
                    iter.next().ok_or("--render-wav requires a path")?,
                ));
                continue;
            }
            "--lfe-fold-gain" | "--target-i" | "--target-tp" | "--target-lra" => iter
                .next()
                .ok_or_else(|| format!("{arg} requires a value"))?,
            "--help" | "-h" => {
                println!(
                    "Usage: stems-upmixer-loudnorm --bed BED.wav [--lfe-fold-gain N] [--target-i N] [--target-tp N] [--target-lra N] [--analysis-json PATH] [--analysis-svg PATH] [--render-wav PATH]"
                );
                std::process::exit(0);
            }
            _ => return Err(format!("unknown argument: {arg}").into()),
        };
        match arg.as_str() {
            "--lfe-fold-gain" => lfe_fold_gain = value.parse()?,
            "--target-i" => target_i = value.parse()?,
            "--target-tp" => target_tp = value.parse()?,
            "--target-lra" => target_lra = value.parse()?,
            _ => unreachable!(),
        }
    }
    Ok(Args {
        bed: bed.ok_or("--bed is required")?,
        lfe_fold_gain,
        target_i,
        target_tp,
        target_lra,
        analysis_json,
        analysis_svg,
        render_wav,
    })
}

fn frame_size(sample_rate: u32, frame_len_msec: u32) -> usize {
    let size = ((sample_rate as f64) * (frame_len_msec as f64 / 1000.0)).round() as usize;
    size + (size % 2)
}

fn process_frame(
    frame: &[f64; 12],
    filters: &mut [KWeightChannel; 5],
    lfe_fold_gain: f64,
    peak: &mut f64,
    power: &mut Vec<f64>,
) {
    let folded = fold_714_to_51(frame, lfe_fold_gain);
    for sample in folded {
        *peak = peak.max(sample.abs());
    }
    let weighted = [
        filters[0].process(folded[0]),
        filters[1].process(folded[1]),
        filters[2].process(folded[2]),
        filters[3].process(folded[4]),
        filters[4].process(folded[5]),
    ];
    power.push(
        weighted[0] * weighted[0]
            + weighted[1] * weighted[1]
            + weighted[2] * weighted[2]
            + 1.41 * weighted[3] * weighted[3]
            + 1.41 * weighted[4] * weighted[4],
    );
}

fn measure_wav(path: &PathBuf, lfe_fold_gain: f64) -> Result<(Vec<f64>, f64, u32), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels < 12 {
        return Err(format!("expected at least 12 channels, got {}", spec.channels).into());
    }
    let channels = spec.channels as usize;
    let mut filters = [
        KWeightChannel::new(spec.sample_rate),
        KWeightChannel::new(spec.sample_rate),
        KWeightChannel::new(spec.sample_rate),
        KWeightChannel::new(spec.sample_rate),
        KWeightChannel::new(spec.sample_rate),
    ];
    let duration = reader.duration() as usize / channels;
    let mut power = Vec::with_capacity(duration);
    let mut peak = 0.0_f64;
    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut current = [0.0_f64; 12];
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64;
                }
                if channel == channels - 1 {
                    process_frame(&current, &mut filters, lfe_fold_gain, &mut peak, &mut power);
                    current = [0.0; 12];
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi((spec.bits_per_sample as i32) - 1);
            let mut current = [0.0_f64; 12];
            for (index, sample) in reader.samples::<i32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64 / scale;
                }
                if channel == channels - 1 {
                    process_frame(&current, &mut filters, lfe_fold_gain, &mut peak, &mut power);
                    current = [0.0; 12];
                }
            }
        }
    }
    Ok((power, peak, spec.sample_rate))
}

#[derive(Clone, Copy, PartialEq)]
enum LimiterPhase {
    Out,
    Attack,
    Sustain,
    Release,
}

struct PeakLimiter {
    buf: Vec<f64>,
    channels: usize,
    write_index: usize,
    env_index: usize,
    peak_index: usize,
    env_cnt: usize,
    attack_length: usize,
    default_attack_length: usize,
    release_length: usize,
    lookahead: usize,
    state: LimiterPhase,
    gain_reduction: [f64; 2],
    prev_smp: Vec<f64>,
    ceiling: f64,
    first_block: bool,
}

impl PeakLimiter {
    fn new(sample_rate: u32, channels: usize, ceiling: f64) -> Self {
        let default_attack_length = frame_size(sample_rate, 10);
        Self {
            buf: vec![0.0; frame_size(sample_rate, 210) * channels],
            channels,
            write_index: 0,
            env_index: 0,
            peak_index: 0,
            env_cnt: 0,
            attack_length: default_attack_length,
            default_attack_length,
            release_length: frame_size(sample_rate, 100),
            lookahead: default_attack_length,
            state: LimiterPhase::Out,
            gain_reduction: [1.0, 1.0],
            prev_smp: vec![0.0; channels],
            ceiling,
            first_block: true,
        }
    }

    fn wrap_index(&self, index: isize) -> usize {
        let size = self.buf.len() as isize;
        index.rem_euclid(size) as usize
    }

    fn read_abs(&self, index: isize) -> f64 {
        self.buf[self.wrap_index(index)].abs()
    }

    fn feed(&mut self, samples: &[f64]) {
        debug_assert_eq!(samples.len(), self.channels);
        for value in samples {
            self.buf[self.write_index] = *value;
            self.write_index += 1;
            if self.write_index >= self.buf.len() {
                self.write_index = 0;
            }
        }
    }

    fn feed_silence(&mut self, frames: usize) {
        let zeros = vec![0.0; self.channels];
        for _ in 0..frames {
            self.feed(&zeros);
        }
    }

    fn detect_peak(&mut self, offset: isize, nb_samples: isize) -> Option<(isize, f64)> {
        let channels = self.channels as isize;
        let mut index =
            self.write_index as isize + (offset * channels) + (self.lookahead as isize * channels);

        if self.first_block {
            for channel in 0..self.channels {
                self.prev_smp[channel] = self.read_abs(index + channel as isize - channels);
            }
        }

        for n in 0..nb_samples {
            for channel in 0..self.channels {
                let channel_offset = channel as isize;
                let this = self.read_abs(index + channel_offset);
                let next = self.read_abs(index + channel_offset + channels);

                if self.prev_smp[channel] <= this && next <= this && this > self.ceiling && n > 0 {
                    let mut detected = true;
                    for lookahead in 2..12 {
                        let next =
                            self.read_abs(index + channel_offset + (lookahead as isize * channels));
                        if next > this {
                            detected = false;
                            break;
                        }
                    }

                    if !detected {
                        continue;
                    }

                    let mut max_peak = 0.0_f64;
                    for peak_channel in 0..self.channels {
                        let peak = self.read_abs(index + peak_channel as isize);
                        if peak_channel == 0 || peak > max_peak {
                            max_peak = peak;
                        }
                        self.prev_smp[peak_channel] = self.read_abs(index + peak_channel as isize);
                    }

                    self.peak_index = self.wrap_index(index);
                    return Some((n, max_peak));
                }

                self.prev_smp[channel] = this;
            }

            index += channels;
        }

        None
    }

    fn apply_gain_at_env_index(&mut self, gain: f64) {
        for channel in 0..self.channels {
            let index = self.wrap_index(self.env_index as isize + channel as isize);
            self.buf[index] *= gain;
        }
        self.env_index += self.channels;
        if self.env_index >= self.buf.len() {
            self.env_index -= self.buf.len();
        }
    }

    fn process_block<F>(&mut self, nb_samples: usize, mut emit: F)
    where
        F: FnMut(&[f64]),
    {
        let channels = self.channels as isize;
        let mut output_index = self.write_index;
        let mut smp_cnt = 0isize;
        let nb_samples = nb_samples as isize;

        if self.first_block {
            let mut max_peak = 0.0_f64;
            for frame in 0..self.lookahead {
                for channel in 0..self.channels {
                    max_peak = max_peak.max(self.buf[frame * self.channels + channel].abs());
                }
            }

            if max_peak > self.ceiling {
                self.gain_reduction[1] = self.ceiling / max_peak;
                self.state = LimiterPhase::Sustain;
                for frame in 0..self.lookahead {
                    for channel in 0..self.channels {
                        self.buf[frame * self.channels + channel] *= self.gain_reduction[1];
                    }
                }
            }
        }

        while smp_cnt < nb_samples {
            match self.state {
                LimiterPhase::Out => {
                    if let Some((peak_delta, peak_value)) =
                        self.detect_peak(smp_cnt, nb_samples - smp_cnt)
                    {
                        self.env_cnt = 0;
                        smp_cnt += peak_delta - self.attack_length as isize;
                        self.gain_reduction[0] = 1.0;
                        self.gain_reduction[1] = self.ceiling / peak_value;
                        self.state = LimiterPhase::Attack;

                        self.env_index = self.wrap_index(
                            self.peak_index as isize - self.attack_length as isize * channels,
                        );
                    } else {
                        smp_cnt = nb_samples;
                    }
                }
                LimiterPhase::Attack => {
                    while self.env_cnt < self.attack_length {
                        let env = self.gain_reduction[0]
                            - (self.env_cnt as f64 / (self.attack_length - 1) as f64)
                                * (self.gain_reduction[0] - self.gain_reduction[1]);
                        self.apply_gain_at_env_index(env);
                        smp_cnt += 1;
                        self.env_cnt += 1;
                        if smp_cnt >= nb_samples {
                            break;
                        }
                    }

                    if smp_cnt < nb_samples {
                        self.env_cnt = 0;
                        self.attack_length = self.default_attack_length;
                        self.state = LimiterPhase::Sustain;
                    }
                }
                LimiterPhase::Sustain => {
                    if let Some((peak_delta, peak_value)) = self.detect_peak(smp_cnt, nb_samples) {
                        let gain_reduction = self.ceiling / peak_value;

                        if gain_reduction < self.gain_reduction[1] {
                            self.state = LimiterPhase::Attack;
                            self.attack_length = peak_delta.max(2) as usize;
                            self.gain_reduction[0] = self.gain_reduction[1];
                            self.gain_reduction[1] = gain_reduction;
                            self.env_cnt = 0;
                            continue;
                        }

                        self.env_cnt = 0;
                        while self.env_cnt < peak_delta as usize {
                            self.apply_gain_at_env_index(self.gain_reduction[1]);
                            smp_cnt += 1;
                            self.env_cnt += 1;
                            if smp_cnt >= nb_samples {
                                break;
                            }
                        }
                    } else {
                        self.state = LimiterPhase::Release;
                        self.gain_reduction[0] = self.gain_reduction[1];
                        self.gain_reduction[1] = 1.0;
                        self.env_cnt = 0;
                    }
                }
                LimiterPhase::Release => {
                    while self.env_cnt < self.release_length {
                        let env = self.gain_reduction[0]
                            + (self.env_cnt as f64 / (self.release_length - 1) as f64)
                                * (self.gain_reduction[1] - self.gain_reduction[0]);
                        self.apply_gain_at_env_index(env);
                        smp_cnt += 1;
                        self.env_cnt += 1;
                        if smp_cnt >= nb_samples {
                            break;
                        }
                    }

                    if smp_cnt < nb_samples {
                        self.env_cnt = 0;
                        self.state = LimiterPhase::Out;
                    }
                }
            }
        }

        let mut frame = vec![0.0_f64; self.channels];
        for _ in 0..nb_samples {
            for (channel, sample) in frame.iter_mut().enumerate() {
                let mut value = self.buf[output_index + channel];
                if value.abs() > self.ceiling {
                    value = self.ceiling * if value < 0.0 { -1.0 } else { 1.0 };
                }
                *sample = value;
            }
            emit(&frame);
            output_index += self.channels;
            if output_index >= self.buf.len() {
                output_index -= self.buf.len();
            }
        }

        self.first_block = false;
    }
}

struct OutputMeter {
    filters: [KWeightChannel; 5],
    duration: usize,
    emitted_frames: usize,
    power: Vec<f64>,
    peak: f64,
}

impl OutputMeter {
    fn new(sample_rate: u32, duration: usize) -> Self {
        Self {
            filters: [
                KWeightChannel::new(sample_rate),
                KWeightChannel::new(sample_rate),
                KWeightChannel::new(sample_rate),
                KWeightChannel::new(sample_rate),
                KWeightChannel::new(sample_rate),
            ],
            duration,
            emitted_frames: 0,
            power: Vec::with_capacity(duration),
            peak: 0.0,
        }
    }

    fn measure(&mut self, samples: &[f64]) {
        if self.emitted_frames >= self.duration {
            return;
        }
        for sample in samples {
            self.peak = self.peak.max(sample.abs());
        }
        let weighted = [
            self.filters[0].process(samples[0]),
            self.filters[1].process(samples[1]),
            self.filters[2].process(samples[2]),
            self.filters[3].process(samples[3]),
            self.filters[4].process(samples[4]),
        ];
        self.power.push(
            weighted[0] * weighted[0]
                + weighted[1] * weighted[1]
                + weighted[2] * weighted[2]
                + 1.41 * weighted[3] * weighted[3]
                + 1.41 * weighted[4] * weighted[4],
        );
        self.emitted_frames += 1;
    }
}

struct DynamicFeed<'a> {
    lfe_fold_gain: f64,
    gains: &'a [f64],
    step_len: usize,
    limiter_delay: usize,
    source_frames: usize,
    pending_frames: usize,
}

impl DynamicFeed<'_> {
    fn process_frame(
        &mut self,
        frame_index: usize,
        current: &[f64; 12],
        limiter: &mut PeakLimiter,
        meter: &mut OutputMeter,
    ) {
        let folded = fold_714_to_51(current, self.lfe_fold_gain);
        let block = frame_index / self.step_len;
        let next = (block + 1).min(self.gains.len().saturating_sub(1));
        let position = (frame_index % self.step_len) as f64 / self.step_len as f64;
        let gain = self.gains.get(block).copied().unwrap_or(1.0) * (1.0 - position)
            + self.gains.get(next).copied().unwrap_or(1.0) * position;
        limiter.feed(&[
            folded[0] * gain,
            folded[1] * gain,
            folded[2] * gain,
            folded[4] * gain,
            folded[5] * gain,
        ]);
        self.source_frames += 1;
        if self.source_frames == self.limiter_delay {
            limiter.process_block(self.step_len, |samples| meter.measure(samples));
        } else if self.source_frames > self.limiter_delay {
            self.pending_frames += 1;
            if self.pending_frames >= self.step_len {
                limiter.process_block(self.step_len, |samples| meter.measure(samples));
                self.pending_frames -= self.step_len;
            }
        }
    }
}

fn measure_dynamic_output_simple(
    path: &PathBuf,
    lfe_fold_gain: f64,
    gains: &[f64],
    target_tp: f64,
    sample_rate: u32,
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels < 12 {
        return Err(format!("expected at least 12 channels, got {}", spec.channels).into());
    }
    if spec.sample_rate != sample_rate {
        return Err(format!(
            "sample rate changed while re-reading WAV: {} != {}",
            spec.sample_rate, sample_rate
        )
        .into());
    }
    let channels = spec.channels as usize;
    let step_len = frame_size(sample_rate, 100);
    let ceiling = 10.0_f64.powf(target_tp / 20.0);
    let mut filters = [
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
    ];
    let duration = reader.duration() as usize / channels;
    let mut power = Vec::with_capacity(duration);
    let mut peak = 0.0_f64;

    let mut process = |frame_index: usize, current: &[f64; 12]| {
        let folded = fold_714_to_51(current, lfe_fold_gain);
        let block = frame_index / step_len;
        let next = (block + 1).min(gains.len().saturating_sub(1));
        let position = (frame_index % step_len) as f64 / step_len as f64;
        let gain = gains.get(block).copied().unwrap_or(1.0) * (1.0 - position)
            + gains.get(next).copied().unwrap_or(1.0) * position;
        let output = [
            (folded[0] * gain).clamp(-ceiling, ceiling),
            (folded[1] * gain).clamp(-ceiling, ceiling),
            (folded[2] * gain).clamp(-ceiling, ceiling),
            (folded[4] * gain).clamp(-ceiling, ceiling),
            (folded[5] * gain).clamp(-ceiling, ceiling),
        ];
        for sample in output {
            peak = peak.max(sample.abs());
        }
        let weighted = [
            filters[0].process(output[0]),
            filters[1].process(output[1]),
            filters[2].process(output[2]),
            filters[3].process(output[3]),
            filters[4].process(output[4]),
        ];
        power.push(
            weighted[0] * weighted[0]
                + weighted[1] * weighted[1]
                + weighted[2] * weighted[2]
                + 1.41 * weighted[3] * weighted[3]
                + 1.41 * weighted[4] * weighted[4],
        );
    };

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64;
                }
                if channel == channels - 1 {
                    process(frame_index, &current);
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi((spec.bits_per_sample as i32) - 1);
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<i32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64 / scale;
                }
                if channel == channels - 1 {
                    process(frame_index, &current);
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
    }
    Ok((power, peak))
}

fn interpolated_gain(gains: &[f64], frame_index: usize, step_len: usize) -> f64 {
    let block = frame_index / step_len;
    let next = (block + 1).min(gains.len().saturating_sub(1));
    let position = (frame_index % step_len) as f64 / step_len as f64;
    gains.get(block).copied().unwrap_or(1.0) * (1.0 - position)
        + gains.get(next).copied().unwrap_or(1.0) * position
}

fn measure_rendered_51_frame(
    samples: &[f64; 6],
    filters: &mut [KWeightChannel; 5],
    peak: &mut f64,
    power: &mut Vec<f64>,
) {
    for sample in samples {
        *peak = peak.max(sample.abs());
    }
    let weighted = [
        filters[0].process(samples[0]),
        filters[1].process(samples[1]),
        filters[2].process(samples[2]),
        filters[3].process(samples[4]),
        filters[4].process(samples[5]),
    ];
    power.push(
        weighted[0] * weighted[0]
            + weighted[1] * weighted[1]
            + weighted[2] * weighted[2]
            + 1.41 * weighted[3] * weighted[3]
            + 1.41 * weighted[4] * weighted[4],
    );
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

fn write_pcm24_frame<W: Write + Seek>(
    writer: &mut hound::WavWriter<W>,
    samples: &[f64; 6],
) -> Result<(), hound::Error> {
    for sample in samples {
        writer.write_sample(pcm24_sample(*sample))?;
    }
    Ok(())
}

fn render_dynamic_wav_simple(
    path: &PathBuf,
    output_path: &Path,
    lfe_fold_gain: f64,
    gains: &[f64],
    target_tp: f64,
    target_offset: f64,
    sample_rate: u32,
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels < 12 {
        return Err(format!("expected at least 12 channels, got {}", spec.channels).into());
    }
    if spec.sample_rate != sample_rate {
        return Err(format!(
            "sample rate changed while re-reading WAV: {} != {}",
            spec.sample_rate, sample_rate
        )
        .into());
    }
    if let Some(parent) = output_path.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent)?;
        }
    }
    let channels = spec.channels as usize;
    let step_len = frame_size(sample_rate, 100);
    let ceiling = 10.0_f64.powf((target_tp - RENDER_TRUE_PEAK_MARGIN_DB) / 20.0);
    let offset_gain = 10.0_f64.powf(target_offset / 20.0);
    let duration = reader.duration() as usize / channels;
    let mut filters = [
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
        KWeightChannel::new(sample_rate),
    ];
    let mut power = Vec::with_capacity(duration);
    let mut peak = 0.0_f64;
    let wav_spec = hound::WavSpec {
        channels: 6,
        sample_rate,
        bits_per_sample: 24,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(output_path, wav_spec)?;

    let mut process = |frame_index: usize, current: &[f64; 12]| -> Result<(), Box<dyn Error>> {
        let folded = fold_714_to_51(current, lfe_fold_gain);
        let gain = interpolated_gain(gains, frame_index, step_len) * offset_gain;
        let output = [
            (folded[0] * gain).clamp(-ceiling, ceiling),
            (folded[1] * gain).clamp(-ceiling, ceiling),
            (folded[2] * gain).clamp(-ceiling, ceiling),
            (folded[3] * gain).clamp(-ceiling, ceiling),
            (folded[4] * gain).clamp(-ceiling, ceiling),
            (folded[5] * gain).clamp(-ceiling, ceiling),
        ];
        write_pcm24_frame(&mut writer, &output)?;
        measure_rendered_51_frame(&output, &mut filters, &mut peak, &mut power);
        Ok(())
    };

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64;
                }
                if channel == channels - 1 {
                    process(frame_index, &current)?;
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi((spec.bits_per_sample as i32) - 1);
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<i32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64 / scale;
                }
                if channel == channels - 1 {
                    process(frame_index, &current)?;
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
    }
    writer.finalize()?;
    Ok((power, peak))
}

fn measure_dynamic_output(
    path: &PathBuf,
    lfe_fold_gain: f64,
    gains: &[f64],
    target_tp: f64,
    sample_rate: u32,
) -> Result<(Vec<f64>, f64), Box<dyn Error>> {
    if env::var_os("STEMS_UPMIXER_EXPERIMENTAL_LOUDNORM_LIMITER").is_none() {
        return measure_dynamic_output_simple(path, lfe_fold_gain, gains, target_tp, sample_rate);
    }

    let mut reader = hound::WavReader::open(path)?;
    let spec = reader.spec();
    if spec.channels < 12 {
        return Err(format!("expected at least 12 channels, got {}", spec.channels).into());
    }
    if spec.sample_rate != sample_rate {
        return Err(format!(
            "sample rate changed while re-reading WAV: {} != {}",
            spec.sample_rate, sample_rate
        )
        .into());
    }
    let channels = spec.channels as usize;
    let step_len = frame_size(sample_rate, 100);
    let ceiling = 10.0_f64.powf(target_tp / 20.0);
    let duration = reader.duration() as usize / channels;
    let mut limiter = PeakLimiter::new(sample_rate, 5, ceiling);
    let limiter_delay = limiter.buf.len() / limiter.channels;
    let mut feed = DynamicFeed {
        lfe_fold_gain,
        gains,
        step_len,
        limiter_delay,
        source_frames: 0,
        pending_frames: 0,
    };
    let mut meter = OutputMeter::new(sample_rate, duration);

    match spec.sample_format {
        hound::SampleFormat::Float => {
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<f32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64;
                }
                if channel == channels - 1 {
                    feed.process_frame(frame_index, &current, &mut limiter, &mut meter);
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
        hound::SampleFormat::Int => {
            let scale = 2.0_f64.powi((spec.bits_per_sample as i32) - 1);
            let mut current = [0.0_f64; 12];
            let mut frame_index = 0usize;
            for (index, sample) in reader.samples::<i32>().enumerate() {
                let channel = index % channels;
                if channel < 12 {
                    current[channel] = sample? as f64 / scale;
                }
                if channel == channels - 1 {
                    feed.process_frame(frame_index, &current, &mut limiter, &mut meter);
                    frame_index += 1;
                    current = [0.0; 12];
                }
            }
        }
    }
    while meter.emitted_frames < duration {
        let remaining = duration - meter.emitted_frames;
        let block_len = step_len.min(remaining);
        limiter.feed_silence(block_len);
        limiter.process_block(block_len, |samples| meter.measure(samples));
    }
    Ok((meter.power, meter.peak))
}

fn fold_714_to_51(input: &[f64; 12], lfe_fold_gain: f64) -> [f64; 6] {
    [
        0.92 * input[0] + 0.24 * input[2] + 0.20 * input[4] + 0.18 * input[8] + 0.10 * input[10],
        0.92 * input[1] + 0.24 * input[2] + 0.20 * input[5] + 0.18 * input[9] + 0.10 * input[11],
        0.78 * input[2] + 0.05 * input[0] + 0.05 * input[1],
        lfe_fold_gain * input[3],
        0.78 * input[6] + 0.40 * input[4] + 0.20 * input[8] + 0.34 * input[10],
        0.78 * input[7] + 0.40 * input[5] + 0.20 * input[9] + 0.34 * input[11],
    ]
}

fn loudness_from_energy(energy: f64) -> f64 {
    -0.691 + 10.0 * energy.max(1.0e-300).log10()
}

fn block_energies(
    power: &[f64],
    sample_rate: u32,
    block_sec: f64,
    overlap: f64,
    append_silence_sec: f64,
) -> Vec<f64> {
    let block_len = (block_sec * sample_rate as f64).round() as usize;
    let step_len = (block_sec * (1.0 - overlap) * sample_rate as f64).round() as usize;
    let total_len = power.len() + (append_silence_sec * sample_rate as f64).round() as usize;
    if block_len == 0 || step_len == 0 || total_len < block_len {
        return Vec::new();
    }
    let block_count = (((total_len as f64 / sample_rate as f64 - block_sec)
        / (block_sec * (1.0 - overlap)))
        .round() as isize
        + 1)
    .max(0) as usize;
    let mut cumulative = Vec::with_capacity(power.len() + 1);
    cumulative.push(0.0);
    let mut sum = 0.0;
    for value in power {
        sum += *value;
        cumulative.push(sum);
    }
    let mut energies = Vec::with_capacity(block_count);
    for block in 0..block_count {
        let start = block * step_len;
        let end = start + block_len;
        let clamped_start = start.min(power.len());
        let clamped_end = end.min(power.len());
        let block_sum = cumulative[clamped_end] - cumulative[clamped_start];
        energies.push(block_sum / block_len as f64);
    }
    energies
}

fn integrated_loudness(power: &[f64], sample_rate: u32) -> (f64, f64) {
    let energies = block_energies(power, sample_rate, 0.400, 0.75, 0.0);
    integrated_loudness_from_energies(&energies)
}

fn integrated_loudness_from_energies(energies: &[f64]) -> (f64, f64) {
    let loudness: Vec<f64> = energies
        .iter()
        .map(|value| loudness_from_energy(*value))
        .collect();
    let abs_gated: Vec<f64> = energies
        .iter()
        .zip(loudness.iter())
        .filter_map(|(energy, loud)| if *loud >= -70.0 { Some(*energy) } else { None })
        .collect();
    if abs_gated.is_empty() {
        return (-f64::INFINITY, -70.0);
    }
    let abs_mean = abs_gated.iter().sum::<f64>() / abs_gated.len() as f64;
    let threshold = loudness_from_energy(abs_mean) - 10.0;
    let rel_gated: Vec<f64> = energies
        .iter()
        .zip(loudness.iter())
        .filter_map(|(energy, loud)| {
            if *loud > threshold && *loud > -70.0 {
                Some(*energy)
            } else {
                None
            }
        })
        .collect();
    if rel_gated.is_empty() {
        return (-f64::INFINITY, threshold);
    }
    let mean = rel_gated.iter().sum::<f64>() / rel_gated.len() as f64;
    (loudness_from_energy(mean), threshold)
}

fn gaussian_weights() -> [f64; 21] {
    let sigma = 3.5;
    let c1 = 1.0 / (sigma * (2.0 * PI).sqrt());
    let c2 = 2.0 * sigma * sigma;
    let mut weights = [0.0_f64; 21];
    let mut total = 0.0;
    for (index, weight) in weights.iter_mut().enumerate() {
        let x = index as i32 - 10;
        *weight = c1 * (-((x * x) as f64 / c2)).exp();
        total += *weight;
    }
    for weight in &mut weights {
        *weight /= total;
    }
    weights
}

fn gaussian_filter(delta: &[f64; 30], index: usize, weights: &[f64; 21]) -> f64 {
    let base = if index > 10 { index - 10 } else { index + 20 };
    let mut result = 0.0;
    for (offset, weight) in weights.iter().enumerate() {
        result += delta[(base + offset) % 30] * weight;
    }
    result
}

fn shortterm_loudness_blocks(power: &[f64], sample_rate: u32) -> Vec<f64> {
    block_energies(power, sample_rate, 3.0, 29.0 / 30.0, 0.0)
        .into_iter()
        .map(loudness_from_energy)
        .collect()
}

fn dynamic_gain_curve(power: &[f64], sample_rate: u32, target_i: f64, target_lra: f64) -> Vec<f64> {
    let step_len = (sample_rate as f64 * 0.100).round().max(1.0) as usize;
    let block_count = power.len().div_ceil(step_len);
    if block_count == 0 {
        return Vec::new();
    }
    let energies_400 = block_energies(power, sample_rate, 0.400, 0.75, 0.0);
    let shortterm = shortterm_loudness_blocks(power, sample_rate);
    let weights = gaussian_weights();
    let mut delta = [1.0_f64; 30];
    let mut index = 1usize;
    let first_shortterm = shortterm.first().copied().unwrap_or(-70.0);
    let first_env = if first_shortterm <= -70.0 {
        0.0
    } else {
        target_i - first_shortterm
    };
    let initial_delta = 10.0_f64.powf(first_env / 20.0);
    delta.fill(initial_delta);
    let mut prev_delta = delta[index];
    let above_threshold = true;
    let mut gains = Vec::with_capacity(block_count + 1);
    for block in 0..=block_count {
        let gain_index = if index + 10 < 30 {
            index + 10
        } else {
            index + 10 - 30
        };
        gains.push(gaussian_filter(&delta, gain_index, &weights));
        if block >= block_count {
            break;
        }
        let shortterm_index = (block + 1).min(shortterm.len().saturating_sub(1));
        let current_shortterm = shortterm
            .get(shortterm_index)
            .copied()
            .unwrap_or(first_shortterm);
        let energy_count = (block + 31).min(energies_400.len()).max(1);
        let (global, relative_threshold) =
            integrated_loudness_from_energies(&energies_400[..energy_count]);
        if current_shortterm < relative_threshold || current_shortterm <= -70.0 || !above_threshold
        {
            delta[index] = prev_delta;
        } else {
            let half_lra = target_lra / 2.0;
            let env_global = (current_shortterm - global).clamp(-half_lra, half_lra);
            let env_shortterm = target_i - current_shortterm;
            delta[index] = 10.0_f64.powf((env_global + env_shortterm) / 20.0);
        }
        prev_delta = delta[index];
        index += 1;
        if index >= 30 {
            index -= 30;
        }
    }
    gains
}

fn ebu_percentile(sorted_values: &[f64], percent: f64) -> f64 {
    if sorted_values.is_empty() {
        return f64::NAN;
    }
    let index = ((sorted_values.len() - 1) as f64 * percent / 100.0).round() as usize;
    sorted_values[index.min(sorted_values.len() - 1)]
}

fn shortterm_loudness_values(power: &[f64], sample_rate: u32) -> Vec<f64> {
    block_energies(power, sample_rate, 3.0, 29.0 / 30.0, 0.0)
        .iter()
        .map(|value| loudness_from_energy(*value))
        .collect()
}

fn loudness_range_from_shortterm(shortterm_loudness: &[f64]) -> f64 {
    let abs_gated: Vec<f64> = shortterm_loudness
        .iter()
        .copied()
        .filter(|value| *value >= -70.0)
        .collect();
    if abs_gated.is_empty() {
        return f64::NAN;
    }
    let stl_power = abs_gated
        .iter()
        .map(|value| 10.0_f64.powf(value / 10.0))
        .sum::<f64>()
        / abs_gated.len() as f64;
    let threshold = 10.0 * stl_power.max(1.0e-300).log10() - 20.0;
    let mut rel_gated: Vec<f64> = abs_gated
        .into_iter()
        .filter(|value| *value >= threshold)
        .collect();
    if rel_gated.is_empty() {
        return f64::NAN;
    }
    rel_gated.sort_by(|left, right| left.partial_cmp(right).unwrap());
    ebu_percentile(&rel_gated, 95.0) - ebu_percentile(&rel_gated, 10.0)
}

fn loudness_range(power: &[f64], sample_rate: u32) -> f64 {
    let shortterm_loudness = shortterm_loudness_values(power, sample_rate);
    loudness_range_from_shortterm(&shortterm_loudness)
}

#[derive(Clone)]
struct ShortTermStats {
    absolute_gate_lufs: f64,
    relative_gate_lufs: f64,
    abs_gated_count: usize,
    rel_gated_count: usize,
    total_count: usize,
    min_lufs: f64,
    max_lufs: f64,
    mean_lufs: f64,
    std_lu: f64,
    p10_lufs: f64,
    p25_lufs: f64,
    p50_lufs: f64,
    p75_lufs: f64,
    p95_lufs: f64,
    lra_lu: f64,
    core_lra_lu: f64,
    volatility_lu_per_sec: f64,
}

struct ShortTermAnalysis {
    values: Vec<f64>,
    stats: ShortTermStats,
}

struct AuditionMarker {
    kind: &'static str,
    center_sec: f64,
    lufs: f64,
    delta_lu: f64,
}

fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

fn stddev(values: &[f64], mean: f64) -> f64 {
    if values.len() < 2 || !mean.is_finite() {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| {
            let delta = value - mean;
            delta * delta
        })
        .sum::<f64>()
        / values.len() as f64;
    variance.sqrt()
}

fn median_sorted(sorted_values: &[f64]) -> f64 {
    ebu_percentile(sorted_values, 50.0)
}

fn volatility_lu_per_sec(values: &[f64], hop_sec: f64) -> f64 {
    if values.len() < 2 {
        return 0.0;
    }
    let mut deltas: Vec<f64> = values
        .windows(2)
        .map(|window| (window[1] - window[0]).abs() / hop_sec)
        .filter(|value| value.is_finite())
        .collect();
    if deltas.is_empty() {
        return 0.0;
    }
    deltas.sort_by(|left, right| left.partial_cmp(right).unwrap());
    median_sorted(&deltas)
}

fn shortterm_analysis(power: &[f64], sample_rate: u32) -> ShortTermAnalysis {
    let values = shortterm_loudness_values(power, sample_rate);
    let abs_gated: Vec<f64> = values
        .iter()
        .copied()
        .filter(|value| *value >= -70.0)
        .collect();
    let absolute_gate_lufs = -70.0;
    let relative_gate_lufs = if abs_gated.is_empty() {
        f64::NAN
    } else {
        let stl_power = abs_gated
            .iter()
            .map(|value| 10.0_f64.powf(value / 10.0))
            .sum::<f64>()
            / abs_gated.len() as f64;
        10.0 * stl_power.max(1.0e-300).log10() - 20.0
    };
    let mut rel_gated: Vec<f64> = abs_gated
        .into_iter()
        .filter(|value| *value >= relative_gate_lufs)
        .collect();
    rel_gated.sort_by(|left, right| left.partial_cmp(right).unwrap());

    let rel_mean = mean(&rel_gated);
    let stats = ShortTermStats {
        absolute_gate_lufs,
        relative_gate_lufs,
        abs_gated_count: values
            .iter()
            .filter(|value| **value >= absolute_gate_lufs)
            .count(),
        rel_gated_count: rel_gated.len(),
        total_count: values.len(),
        min_lufs: rel_gated.first().copied().unwrap_or(f64::NAN),
        max_lufs: rel_gated.last().copied().unwrap_or(f64::NAN),
        mean_lufs: rel_mean,
        std_lu: stddev(&rel_gated, rel_mean),
        p10_lufs: ebu_percentile(&rel_gated, 10.0),
        p25_lufs: ebu_percentile(&rel_gated, 25.0),
        p50_lufs: ebu_percentile(&rel_gated, 50.0),
        p75_lufs: ebu_percentile(&rel_gated, 75.0),
        p95_lufs: ebu_percentile(&rel_gated, 95.0),
        lra_lu: loudness_range_from_shortterm(&values),
        core_lra_lu: ebu_percentile(&rel_gated, 75.0) - ebu_percentile(&rel_gated, 25.0),
        volatility_lu_per_sec: volatility_lu_per_sec(&values, 0.1),
    };

    ShortTermAnalysis { values, stats }
}

fn audition_markers(analysis: &ShortTermAnalysis) -> Vec<AuditionMarker> {
    let mut markers = Vec::new();
    let hop_sec = 0.1;
    let center_for_index = |index: usize| index as f64 * hop_sec + 1.5;

    if let Some((index, value)) = analysis
        .values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .min_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
    {
        markers.push(AuditionMarker {
            kind: "quietest_shortterm",
            center_sec: center_for_index(index),
            lufs: *value,
            delta_lu: 0.0,
        });
    }
    if let Some((index, value)) = analysis
        .values
        .iter()
        .enumerate()
        .filter(|(_, value)| value.is_finite())
        .max_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap())
    {
        markers.push(AuditionMarker {
            kind: "loudest_shortterm",
            center_sec: center_for_index(index),
            lufs: *value,
            delta_lu: 0.0,
        });
    }

    if analysis.values.len() >= 11 {
        let mut changes: Vec<(usize, f64)> = analysis
            .values
            .windows(11)
            .enumerate()
            .filter_map(|(index, window)| {
                let start = window.first().copied()?;
                let end = window.last().copied()?;
                if start.is_finite() && end.is_finite() {
                    Some((index + 10, end - start))
                } else {
                    None
                }
            })
            .collect();
        changes.sort_by(|(_, left), (_, right)| right.partial_cmp(left).unwrap());
        if let Some((index, delta)) = changes.first() {
            markers.push(AuditionMarker {
                kind: "largest_1s_rise",
                center_sec: center_for_index(*index),
                lufs: analysis.values[*index],
                delta_lu: *delta,
            });
        }
        changes.sort_by(|(_, left), (_, right)| left.partial_cmp(right).unwrap());
        if let Some((index, delta)) = changes.first() {
            markers.push(AuditionMarker {
                kind: "largest_1s_drop",
                center_sec: center_for_index(*index),
                lufs: analysis.values[*index],
                delta_lu: *delta,
            });
        }
    }

    markers
}

fn json_escape(value: &str) -> String {
    let mut escaped = String::with_capacity(value.len() + 8);
    for ch in value.chars() {
        match ch {
            '"' => escaped.push_str("\\\""),
            '\\' => escaped.push_str("\\\\"),
            '\n' => escaped.push_str("\\n"),
            '\r' => escaped.push_str("\\r"),
            '\t' => escaped.push_str("\\t"),
            _ => escaped.push(ch),
        }
    }
    escaped
}

fn json_number(value: f64) -> String {
    if value.is_finite() {
        format!("{value:.6}")
    } else {
        "null".to_string()
    }
}

fn append_stats_json(out: &mut String, stats: &ShortTermStats) {
    out.push_str(&format!(
        "\"absolute_gate_lufs\":{},\"relative_gate_lufs\":{},\"abs_gated_count\":{},\"rel_gated_count\":{},\"total_count\":{},\"min_lufs\":{},\"max_lufs\":{},\"mean_lufs\":{},\"std_lu\":{},\"p10_lufs\":{},\"p25_lufs\":{},\"p50_lufs\":{},\"p75_lufs\":{},\"p95_lufs\":{},\"lra_lu\":{},\"core_lra_lu\":{},\"volatility_lu_per_sec\":{}",
        json_number(stats.absolute_gate_lufs),
        json_number(stats.relative_gate_lufs),
        stats.abs_gated_count,
        stats.rel_gated_count,
        stats.total_count,
        json_number(stats.min_lufs),
        json_number(stats.max_lufs),
        json_number(stats.mean_lufs),
        json_number(stats.std_lu),
        json_number(stats.p10_lufs),
        json_number(stats.p25_lufs),
        json_number(stats.p50_lufs),
        json_number(stats.p75_lufs),
        json_number(stats.p95_lufs),
        json_number(stats.lra_lu),
        json_number(stats.core_lra_lu),
        json_number(stats.volatility_lu_per_sec),
    ));
}

fn append_timeline_json(out: &mut String, analysis: &ShortTermAnalysis) {
    out.push('[');
    for (index, value) in analysis.values.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        let start_sec = index as f64 * 0.1;
        let center_sec = start_sec + 1.5;
        let above_abs = *value >= analysis.stats.absolute_gate_lufs;
        let above_rel = above_abs && *value >= analysis.stats.relative_gate_lufs;
        out.push_str(&format!(
            "{{\"index\":{},\"start_sec\":{},\"center_sec\":{},\"lufs\":{},\"above_absolute_gate\":{},\"above_lra_gate\":{}}}",
            index,
            json_number(start_sec),
            json_number(center_sec),
            json_number(*value),
            above_abs,
            above_rel
        ));
    }
    out.push(']');
}

fn append_markers_json(out: &mut String, analysis: &ShortTermAnalysis) {
    let markers = audition_markers(analysis);
    out.push('[');
    for (index, marker) in markers.iter().enumerate() {
        if index > 0 {
            out.push(',');
        }
        out.push_str(&format!(
            "{{\"kind\":\"{}\",\"center_sec\":{},\"lufs\":{},\"delta_lu\":{}}}",
            marker.kind,
            json_number(marker.center_sec),
            json_number(marker.lufs),
            json_number(marker.delta_lu)
        ));
    }
    out.push(']');
}

fn write_analysis_json(
    path: &Path,
    bed_path: &Path,
    sample_rate: u32,
    input_analysis: &ShortTermAnalysis,
    output_analysis: &ShortTermAnalysis,
) -> Result<(), Box<dyn Error>> {
    let mut json = String::new();
    json.push_str("{\"schema\":\"stems-upmixer-shortterm-v1\",");
    json.push_str(&format!(
        "\"bed\":\"{}\",\"sample_rate\":{},\"window_sec\":3.0,\"hop_sec\":0.1,",
        json_escape(&bed_path.display().to_string()),
        sample_rate
    ));
    json.push_str("\"input\":{\"stats\":{");
    append_stats_json(&mut json, &input_analysis.stats);
    json.push_str("},\"audition_markers\":");
    append_markers_json(&mut json, input_analysis);
    json.push_str(",\"timeline\":");
    append_timeline_json(&mut json, input_analysis);
    json.push_str("},\"dynamic_output\":{\"stats\":{");
    append_stats_json(&mut json, &output_analysis.stats);
    json.push_str("},\"audition_markers\":");
    append_markers_json(&mut json, output_analysis);
    json.push_str(",\"timeline\":");
    append_timeline_json(&mut json, output_analysis);
    json.push_str("}}");
    fs::write(path, json)?;
    Ok(())
}

fn svg_path(values: &[f64], x_for: &dyn Fn(usize) -> f64, y_for: &dyn Fn(f64) -> f64) -> String {
    let mut path = String::new();
    let mut started = false;
    for (index, value) in values.iter().enumerate() {
        if !value.is_finite() {
            continue;
        }
        if started {
            path.push('L');
        } else {
            path.push('M');
            started = true;
        }
        path.push_str(&format!("{:.2},{:.2}", x_for(index), y_for(*value)));
    }
    path
}

fn write_analysis_svg(
    path: &Path,
    input_analysis: &ShortTermAnalysis,
    output_analysis: &ShortTermAnalysis,
) -> Result<(), Box<dyn Error>> {
    let width = 1280.0;
    let height = 720.0;
    let left = 72.0;
    let right = 32.0;
    let top = 48.0;
    let bottom = 64.0;
    let chart_w = width - left - right;
    let chart_h = height - top - bottom;
    let max_len = input_analysis
        .values
        .len()
        .max(output_analysis.values.len())
        .max(1);
    let max_time = (max_len - 1) as f64 * 0.1 + 1.5;
    let finite_values: Vec<f64> = input_analysis
        .values
        .iter()
        .chain(output_analysis.values.iter())
        .copied()
        .filter(|value| value.is_finite())
        .collect();
    let raw_min = finite_values
        .iter()
        .copied()
        .fold(f64::INFINITY, |acc, value| acc.min(value));
    let raw_max = finite_values
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, |acc, value| acc.max(value));
    let y_min = ((raw_min - 2.0) / 5.0).floor() * 5.0;
    let y_max = ((raw_max + 2.0) / 5.0).ceil() * 5.0;
    let x_for = |index: usize| left + ((index as f64 * 0.1 + 1.5) / max_time) * chart_w;
    let y_for = |lufs: f64| top + ((y_max - lufs) / (y_max - y_min).max(1.0)) * chart_h;
    let input_path = svg_path(&input_analysis.values, &x_for, &y_for);
    let output_path = svg_path(&output_analysis.values, &x_for, &y_for);

    let mut svg = String::new();
    svg.push_str(&format!(
        "<svg xmlns=\"http://www.w3.org/2000/svg\" viewBox=\"0 0 {width:.0} {height:.0}\" width=\"{width:.0}\" height=\"{height:.0}\">"
    ));
    svg.push_str("<rect width=\"100%\" height=\"100%\" fill=\"#f7f7f2\"/>");
    svg.push_str("<style>text{font-family:Inter,Arial,sans-serif;fill:#262626}.axis{stroke:#9a9a90;stroke-width:1}.grid{stroke:#d8d6ca;stroke-width:1}.gate{stroke-dasharray:6 5}.label{font-size:13px}.title{font-size:20px;font-weight:700}.small{font-size:12px}</style>");
    svg.push_str("<text x=\"72\" y=\"30\" class=\"title\">Short-term loudness vector</text>");
    svg.push_str(&format!(
        "<rect x=\"{left}\" y=\"{top}\" width=\"{chart_w}\" height=\"{chart_h}\" fill=\"#ffffff\" stroke=\"#b8b5a8\"/>"
    ));

    let mut tick = (y_min / 5.0).ceil() * 5.0;
    while tick <= y_max {
        let y = y_for(tick);
        svg.push_str(&format!(
            "<line x1=\"{left}\" x2=\"{}\" y1=\"{y:.2}\" y2=\"{y:.2}\" class=\"grid\"/>",
            left + chart_w
        ));
        svg.push_str(&format!(
            "<text x=\"18\" y=\"{:.2}\" class=\"small\">{tick:.0} LUFS</text>",
            y + 4.0
        ));
        tick += 5.0;
    }

    let time_tick = if max_time > 240.0 { 60.0 } else { 30.0 };
    let mut current_time = 0.0;
    while current_time <= max_time {
        let x = left + (current_time / max_time) * chart_w;
        svg.push_str(&format!(
            "<line x1=\"{x:.2}\" x2=\"{x:.2}\" y1=\"{top}\" y2=\"{}\" class=\"grid\"/>",
            top + chart_h
        ));
        svg.push_str(&format!(
            "<text x=\"{:.2}\" y=\"{}\" class=\"small\">{:.0}s</text>",
            x - 10.0,
            top + chart_h + 24.0,
            current_time
        ));
        current_time += time_tick;
    }

    let mut add_hline = |value: f64, color: &str, label: &str| {
        if value.is_finite() {
            let y = y_for(value);
            svg.push_str(&format!(
                "<line x1=\"{left}\" x2=\"{}\" y1=\"{y:.2}\" y2=\"{y:.2}\" stroke=\"{color}\" stroke-width=\"1.5\" class=\"gate\"/>",
                left + chart_w
            ));
            svg.push_str(&format!(
                "<text x=\"{}\" y=\"{:.2}\" class=\"small\" fill=\"{color}\">{label} {value:.1}</text>",
                left + chart_w - 150.0,
                y - 5.0
            ));
        }
    };
    add_hline(
        input_analysis.stats.relative_gate_lufs,
        "#777777",
        "input gate",
    );
    add_hline(input_analysis.stats.p10_lufs, "#4d7c0f", "input p10");
    add_hline(input_analysis.stats.p95_lufs, "#b45309", "input p95");

    svg.push_str(&format!(
        "<path d=\"{}\" fill=\"none\" stroke=\"#2563eb\" stroke-width=\"2\"/>",
        input_path
    ));
    svg.push_str(&format!(
        "<path d=\"{}\" fill=\"none\" stroke=\"#dc2626\" stroke-width=\"1.8\" opacity=\"0.75\"/>",
        output_path
    ));
    svg.push_str(
        "<rect x=\"86\" y=\"58\" width=\"280\" height=\"72\" fill=\"#ffffff\" stroke=\"#c8c5ba\"/>",
    );
    svg.push_str("<line x1=\"104\" x2=\"150\" y1=\"82\" y2=\"82\" stroke=\"#2563eb\" stroke-width=\"2\"/><text x=\"160\" y=\"87\" class=\"label\">input short-term LUFS</text>");
    svg.push_str("<line x1=\"104\" x2=\"150\" y1=\"108\" y2=\"108\" stroke=\"#dc2626\" stroke-width=\"2\" opacity=\"0.75\"/><text x=\"160\" y=\"113\" class=\"label\">dynamic output estimate</text>");
    svg.push_str(&format!(
        "<text x=\"390\" y=\"86\" class=\"label\">input LRA {:.2} LU / core {:.2} LU / volatility {:.2} LU/s</text>",
        input_analysis.stats.lra_lu,
        input_analysis.stats.core_lra_lu,
        input_analysis.stats.volatility_lu_per_sec
    ));
    svg.push_str(&format!(
        "<text x=\"390\" y=\"112\" class=\"label\">output LRA {:.2} LU / core {:.2} LU / volatility {:.2} LU/s</text>",
        output_analysis.stats.lra_lu,
        output_analysis.stats.core_lra_lu,
        output_analysis.stats.volatility_lu_per_sec
    ));
    svg.push_str("</svg>");
    fs::write(path, svg)?;
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = parse_args()?;
    let timer = Instant::now();
    let (power, sample_peak, sample_rate) = measure_wav(&args.bed, args.lfe_fold_gain)?;
    if sample_rate != 48_000 {
        return Err(format!("expected 48000 Hz bed, got {sample_rate} Hz").into());
    }
    let (input_i, input_thresh) = integrated_loudness(&power, sample_rate);
    let input_lra = loudness_range(&power, sample_rate);
    let input_tp = 20.0 * sample_peak.max(1.0e-300).log10();
    let gains = dynamic_gain_curve(&power, sample_rate, args.target_i, args.target_lra);
    let (output_power, output_peak) = measure_dynamic_output(
        &args.bed,
        args.lfe_fold_gain,
        &gains,
        args.target_tp,
        sample_rate,
    )?;
    let (output_i, output_thresh) = integrated_loudness(&output_power, sample_rate);
    let output_lra = loudness_range(&output_power, sample_rate);
    let output_tp = 20.0 * output_peak.max(1.0e-300).log10();
    let target_offset = args.target_i - output_i;
    let render_metrics = if let Some(path) = args.render_wav.as_deref() {
        let (render_power, render_peak) = render_dynamic_wav_simple(
            &args.bed,
            path,
            args.lfe_fold_gain,
            &gains,
            args.target_tp,
            target_offset,
            sample_rate,
        )?;
        let (render_i, render_thresh) = integrated_loudness(&render_power, sample_rate);
        let render_lra = loudness_range(&render_power, sample_rate);
        let render_tp = 20.0 * render_peak.max(1.0e-300).log10();
        Some((
            path.to_path_buf(),
            render_i,
            render_tp,
            render_lra,
            render_thresh,
        ))
    } else {
        None
    };
    if args.analysis_json.is_some() || args.analysis_svg.is_some() {
        let input_analysis = shortterm_analysis(&power, sample_rate);
        let output_analysis = shortterm_analysis(&output_power, sample_rate);
        if let Some(path) = args.analysis_json.as_deref() {
            write_analysis_json(
                path,
                &args.bed,
                sample_rate,
                &input_analysis,
                &output_analysis,
            )?;
        }
        if let Some(path) = args.analysis_svg.as_deref() {
            write_analysis_svg(path, &input_analysis, &output_analysis)?;
        }
    }
    let mut payload = format!(
        "{{\"input_i\":\"{:.2}\",\"input_tp\":\"{:.2}\",\"input_lra\":\"{:.2}\",\"input_thresh\":\"{:.2}\",\"output_i\":\"{:.2}\",\"output_tp\":\"{:.2}\",\"output_lra\":\"{:.2}\",\"output_thresh\":\"{:.2}\",\"normalization_type\":\"dynamic\",\"target_offset\":\"{:.2}\",\"target_i\":{:.6},\"target_tp\":{:.6},\"target_lra\":{:.6},\"elapsed_sec\":{:.6}",
        input_i,
        input_tp,
        input_lra,
        input_thresh,
        output_i,
        output_tp,
        output_lra,
        output_thresh,
        target_offset,
        args.target_i,
        args.target_tp,
        args.target_lra,
        timer.elapsed().as_secs_f64(),
    );
    if let Some((path, render_i, render_tp, render_lra, render_thresh)) = render_metrics {
        payload.push_str(&format!(
            ",\"render_path\":\"{}\",\"render_i\":\"{:.2}\",\"render_tp\":\"{:.2}\",\"render_lra\":\"{:.2}\",\"render_thresh\":\"{:.2}\"",
            json_escape(&path.display().to_string()),
            render_i,
            render_tp,
            render_lra,
            render_thresh,
        ));
    }
    payload.push('}');
    println!("{payload}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn energy_from_loudness(loudness: f64) -> f64 {
        10.0_f64.powf((loudness + 0.691) / 10.0)
    }

    fn power_from_loudness_segments(
        sample_rate: u32,
        segment_duration_sec: usize,
        loudness_values: &[f64],
    ) -> Vec<f64> {
        let frames_per_segment = sample_rate as usize * segment_duration_sec;
        let mut power = Vec::with_capacity(frames_per_segment * loudness_values.len());
        for loudness in loudness_values {
            power.extend(
                std::iter::repeat(energy_from_loudness(*loudness)).take(frames_per_segment),
            );
        }
        power
    }

    #[test]
    fn ebu_lra_minimum_requirement_tone_cases() {
        let sample_rate = 1000;
        let cases = [
            ([-20.0, -30.0].as_slice(), 10.0),
            ([-20.0, -15.0].as_slice(), 5.0),
            ([-40.0, -20.0].as_slice(), 20.0),
            ([-50.0, -35.0, -20.0, -35.0, -50.0].as_slice(), 15.0),
        ];

        for (segments, expected_lra) in cases {
            let power = power_from_loudness_segments(sample_rate, 20, segments);
            let actual_lra = loudness_range(&power, sample_rate);
            assert!(
                (actual_lra - expected_lra).abs() <= 0.1,
                "segments={segments:?} expected {expected_lra}, got {actual_lra}"
            );
        }
    }

    #[test]
    fn ebu_lra_uses_rounded_percentile_index() {
        let mut shortterm = vec![-20.0; 9];
        shortterm.push(-10.0);

        let lra = loudness_range_from_shortterm(&shortterm);

        assert_eq!(lra, 10.0);
    }
}
