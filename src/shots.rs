//! The shot list: one `[[shot]]` per subject + scene, rendered once per seed.

use serde::Deserialize;
use std::collections::HashSet;

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ShotFile {
    #[serde(default)]
    pub defaults: Defaults,
    #[serde(rename = "shot")]
    pub shots: Vec<Shot>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields, default)]
pub struct Defaults {
    pub seeds: Vec<u64>,
    pub width: u32,
    pub height: u32,
    /// How much of the face crop pass 2 redraws: lower keeps pass 1's face, higher follows `face`.
    pub face_denoise: f32,
}

impl Default for Defaults {
    fn default() -> Self {
        Defaults { seeds: vec![1], width: 1152, height: 1728, face_denoise: 0.5 }
    }
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shot {
    /// Output folder name: lowercase letters, digits, `-`, `_`.
    pub name: String,
    /// Who: age, hair, skin, build, clothes that never change between scenes.
    pub subject: String,
    /// Face details. Pass 1 leaves them out — face phrases pull the camera into a close
    /// selfie framing — and pass 2 redraws only the face with them in.
    #[serde(default)]
    pub face: String,
    /// Where, what they're doing, light, camera, expression.
    pub scene: String,
    pub seeds: Option<Vec<u64>>,
    pub lora: Option<Lora>,
    pub video: Option<Video>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Lora {
    pub name: String,
    #[serde(default = "one")]
    pub strength: f32,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Video {
    /// Keep it small: blinks, breathing, hair in the wind, a still camera. Big head turns
    /// are where the face drifts.
    pub motion: String,
    /// Frames at 16 fps; Wan wants 4k + 1 (49 ≈ 3 s, 81 ≈ 5 s).
    #[serde(default = "default_length")]
    pub length: u32,
    #[serde(default = "default_video_width")]
    pub width: u32,
    #[serde(default = "default_video_height")]
    pub height: u32,
}

fn one() -> f32 {
    1.0
}
fn default_length() -> u32 {
    49
}
fn default_video_width() -> u32 {
    720
}
fn default_video_height() -> u32 {
    1088
}

impl Shot {
    /// (pass 1 prompt, pass 2 prompt): subject + scene, then subject + face + scene.
    pub fn prompts(&self) -> (String, String) {
        let join = |parts: &[&str]| {
            parts.iter().map(|p| p.trim()).filter(|p| !p.is_empty()).collect::<Vec<_>>().join(", ")
        };
        (join(&[&self.subject, &self.scene]), join(&[&self.subject, &self.face, &self.scene]))
    }
}

pub fn parse(text: &str) -> Result<ShotFile, String> {
    let file: ShotFile = toml::from_str(text).map_err(|e| e.to_string())?;
    let d = &file.defaults;
    if d.seeds.is_empty() {
        return Err("defaults.seeds is empty".into());
    }
    check_size("defaults", d.width, d.height)?;
    if !(d.face_denoise > 0.0 && d.face_denoise <= 1.0) {
        return Err(format!("defaults.face_denoise {} is outside (0, 1]", d.face_denoise));
    }
    if file.shots.is_empty() {
        return Err("no [[shot]] entries".into());
    }
    let mut names = HashSet::new();
    for s in &file.shots {
        let ok = !s.name.is_empty()
            && s.name.chars().all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
        if !ok {
            return Err(format!("shot name {:?}: use lowercase letters, digits, '-' and '_'", s.name));
        }
        if !names.insert(&s.name) {
            return Err(format!("shot name {:?} appears twice", s.name));
        }
        if s.seeds.as_ref().is_some_and(Vec::is_empty) {
            return Err(format!("{}: seeds is empty", s.name));
        }
        if let Some(v) = &s.video {
            if v.length % 4 != 1 {
                return Err(format!("{}: video length {} is not 4k + 1 (49, 81, ...)", s.name, v.length));
            }
            check_size(&s.name, v.width, v.height)?;
        }
    }
    Ok(file)
}

fn check_size(what: &str, w: u32, h: u32) -> Result<(), String> {
    if w == 0 || h == 0 || !w.is_multiple_of(16) || !h.is_multiple_of(16) {
        return Err(format!("{what}: size {w}x{h} must be non-zero multiples of 16"));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const MIN: &str = r#"
        [[shot]]
        name = "a"
        subject = "man"
        scene = "beach"
    "#;

    #[test]
    fn pass_one_leaves_the_face_out() {
        let f = parse(&format!("{MIN}face = \"grey eyes\"")).unwrap();
        let (p1, p2) = f.shots[0].prompts();
        assert_eq!(p1, "man, beach");
        assert_eq!(p2, "man, grey eyes, beach");
    }

    #[test]
    fn defaults_fill_in() {
        let f = parse(MIN).unwrap();
        assert_eq!(f.defaults.seeds, vec![1]);
        assert_eq!((f.defaults.width, f.defaults.height), (1152, 1728));
        assert_eq!(f.shots[0].prompts().0, f.shots[0].prompts().1);
    }

    #[test]
    fn rejects_bad_input() {
        for (extra, why) in [
            ("[shot.video]\nmotion = \"x\"\nlength = 50", "4k + 1"),
            ("seeds = []", "seeds is empty"),
            ("typo = 1", "unknown field"),
        ] {
            let err = parse(&format!("{MIN}{extra}")).unwrap_err();
            assert!(err.contains(why), "{extra}: {err}");
        }
        assert!(parse(&MIN.replace("\"a\"", "\"../a\"")).unwrap_err().contains("lowercase"));
        assert!(parse(&format!("{MIN}{MIN}")).unwrap_err().contains("twice"));
        assert!(parse(&format!("[defaults]\nwidth = 1000\n{MIN}")).unwrap_err().contains("multiples of 16"));
    }

    #[test]
    fn example_file_parses() {
        let f = parse(include_str!("../examples/shots.toml")).unwrap();
        assert!(f.shots.iter().any(|s| s.video.is_some()));
    }
}
