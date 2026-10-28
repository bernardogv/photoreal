//! photoreal — render a TOML shot list on a ComfyUI server: each shot × seed becomes a two-pass
//! still (`s<seed>_pass1.png` before the face fix, `s<seed>.png` after) and, when the shot has a
//! `[shot.video]` table, a Wan 2.2 clip from that still (`s<seed>.mp4`). Files already on disk
//! are skipped, so a re-run picks up where the last one stopped.

mod comfy;
mod graphs;
mod shots;

use comfy::Comfy;
use std::fs;
use std::path::Path;
use std::process::ExitCode;

const USAGE: &str = "usage: photoreal render <shots.toml> [--out out] [--comfy http://127.0.0.1:8188]
       photoreal export image|video     print the API-format graph, with placeholder values
env:   PHOTOREAL_COMFY overrides the default server";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let result = match args.first().map(String::as_str) {
        Some("render") => render(&args[1..]),
        Some("export") => export(args.get(1).map(String::as_str)),
        _ => Err(USAGE.into()),
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("photoreal: {e}");
            ExitCode::FAILURE
        }
    }
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).map(String::as_str)
}

fn render(args: &[String]) -> Result<(), String> {
    let path = args.first().filter(|a| !a.starts_with("--")).ok_or(USAGE)?;
    let text = fs::read_to_string(path).map_err(|e| format!("{path}: {e}"))?;
    let file = shots::parse(&text).map_err(|e| format!("{path}: {e}"))?;
    let out = Path::new(flag(args, "--out").unwrap_or("out"));
    let base = flag(args, "--comfy")
        .map(str::to_string)
        .or_else(|| std::env::var("PHOTOREAL_COMFY").ok())
        .unwrap_or_else(|| "http://127.0.0.1:8188".into());
    let comfy = Comfy { base: base.trim_end_matches('/').to_string() };

    let mut failed = 0;
    for shot in &file.shots {
        let dir = out.join(&shot.name);
        fs::create_dir_all(&dir).map_err(|e| format!("{}: {e}", dir.display()))?;
        for &seed in shot.seeds.as_ref().unwrap_or(&file.defaults.seeds) {
            // One failed render (bad model name, server hiccup) shouldn't stop the rest.
            if let Err(e) = render_one(&comfy, &file.defaults, shot, seed, &dir) {
                eprintln!("{} s{seed}: {e}", shot.name);
                failed += 1;
            }
        }
    }
    if failed > 0 {
        return Err(format!("{failed} render(s) failed"));
    }
    Ok(())
}

fn render_one(comfy: &Comfy, d: &shots::Defaults, shot: &shots::Shot, seed: u64, dir: &Path) -> Result<(), String> {
    let still = dir.join(format!("s{seed}.png"));
    let prefix = format!("photoreal/{}_s{seed}", shot.name);
    if still.exists() {
        println!("{} skip (exists)", still.display());
    } else {
        let (pass1, pass2) = shot.prompts();
        let t0 = std::time::Instant::now();
        let outputs = comfy.queue_and_wait(graphs::image(&graphs::ImageJob {
            pass1: &pass1,
            pass2: &pass2,
            seed,
            width: d.width,
            height: d.height,
            face_denoise: d.face_denoise,
            lora: shot.lora.as_ref(),
            prefix: &prefix,
        }))?;
        write(&dir.join(format!("s{seed}_pass1.png")), &comfy.first_output(&outputs, "before")?)?;
        write(&still, &comfy.first_output(&outputs, "save")?)?;
        println!("{} ({:.0}s)", still.display(), t0.elapsed().as_secs_f32());
    }

    let Some(video) = &shot.video else { return Ok(()) };
    let clip = dir.join(format!("s{seed}.mp4"));
    if clip.exists() {
        println!("{} skip (exists)", clip.display());
        return Ok(());
    }
    let png = fs::read(&still).map_err(|e| format!("{}: {e}", still.display()))?;
    let uploaded = comfy.upload(&format!("photoreal_{}_s{seed}.png", shot.name), &png)?;
    let t0 = std::time::Instant::now();
    let outputs = comfy.queue_and_wait(graphs::video(&uploaded, video, seed, &prefix))?;
    write(&clip, &comfy.first_output(&outputs, "save")?)?;
    println!("{} ({:.0}s)", clip.display(), t0.elapsed().as_secs_f32());
    Ok(())
}

fn write(path: &Path, bytes: &[u8]) -> Result<(), String> {
    fs::write(path, bytes).map_err(|e| format!("{}: {e}", path.display()))
}

fn export(which: Option<&str>) -> Result<(), String> {
    let graph = match which {
        Some("image") => graphs::image(&graphs::ImageJob {
            pass1: "SUBJECT, SCENE",
            pass2: "SUBJECT, FACE, SCENE",
            seed: 1,
            width: 1152,
            height: 1728,
            face_denoise: 0.5,
            lora: None,
            prefix: "photoreal",
        }),
        Some("video") => {
            let v = shots::Video { motion: "MOTION".into(), length: 49, width: 720, height: 1088 };
            graphs::video("input.png", &v, 1, "photoreal")
        }
        _ => return Err(USAGE.into()),
    };
    // Serializing a `Value` can't fail: every map key is a string.
    println!("{}", serde_json::to_string_pretty(&graph).expect("Value serializes"));
    Ok(())
}
