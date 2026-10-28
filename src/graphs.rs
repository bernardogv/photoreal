//! The two ComfyUI graphs, in API format.
//!
//! Image: GonzaLomo Krea2 at 8 steps, cfg 1, euler/beta, with a zeroed negative (cfg 1 ignores
//! it anyway). Pass 1 renders the whole frame without face phrases; pass 2 detects the faces,
//! keeps the largest, and redraws only that crop with the full prompt.
//!
//! Video: Wan 2.2 i2v 14B, high-noise → low-noise experts with the lightx2v 4-step LoRAs.

use crate::shots::{Lora, Video};
use serde_json::{Value, json};

pub const FPS: u32 = 16;
const VIDEO_NEGATIVE: &str = "blurry, distorted face, extra fingers, warped, text, watermark, jump cut, camera shake";

pub struct ImageJob<'a> {
    pub pass1: &'a str,
    pub pass2: &'a str,
    pub seed: u64,
    pub width: u32,
    pub height: u32,
    pub face_denoise: f32,
    pub lora: Option<&'a Lora>,
    pub prefix: &'a str,
}

/// Outputs: `before` (pass 1) and `save` (after the face fix).
pub fn image(j: &ImageJob) -> Value {
    let mut g = json!({
        "unet": {"class_type": "UNETLoader",
                 "inputs": {"unet_name": "gonzalomoKrea2_v20_fp8.safetensors", "weight_dtype": "default"}},
        "clip": {"class_type": "CLIPLoader",
                 "inputs": {"clip_name": "qwen3vl_4b_fp8_scaled.safetensors", "type": "krea2", "device": "default"}},
        "vae":  {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "pos1": {"class_type": "CLIPTextEncode", "inputs": {"text": j.pass1, "clip": ["clip", 0]}},
        "neg1": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos1", 0]}},
        "latent": {"class_type": "EmptySD3LatentImage",
                   "inputs": {"width": j.width, "height": j.height, "batch_size": 1}},
        "sample": {"class_type": "KSampler",
                   "inputs": {"seed": j.seed, "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta",
                              "denoise": 1.0, "model": ["unet", 0], "positive": ["pos1", 0],
                              "negative": ["neg1", 0], "latent_image": ["latent", 0]}},
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["sample", 0], "vae": ["vae", 0]}},
        "before": {"class_type": "SaveImage",
                   "inputs": {"filename_prefix": format!("{}_pass1", j.prefix), "images": ["decode", 0]}},
        "pos2": {"class_type": "CLIPTextEncode", "inputs": {"text": j.pass2, "clip": ["clip", 0]}},
        "neg2": {"class_type": "ConditioningZeroOut", "inputs": {"conditioning": ["pos2", 0]}},
        "det": {"class_type": "UltralyticsDetectorProvider", "inputs": {"model_name": "bbox/face_yolov8m.pt"}},
        "faces": {"class_type": "BboxDetectorSEGS", "inputs": {
            "bbox_detector": ["det", 0], "image": ["decode", 0], "threshold": 0.5, "dilation": 10,
            "crop_factor": 3.0, "drop_size": 10, "labels": "all"}},
        // Only the largest face: redrawing every face puts the subject's face on bystanders.
        "largest": {"class_type": "ImpactSEGSOrderedFilter", "inputs": {
            "segs": ["faces", 0], "target": "area(=w*h)", "order": true, "take_start": 0, "take_count": 1}},
        "fix": {"class_type": "DetailerForEach", "inputs": {
            "image": ["decode", 0], "segs": ["largest", 0], "model": ["unet", 0], "clip": ["clip", 0],
            "vae": ["vae", 0], "guide_size": 768, "guide_size_for": true, "max_size": 1024, "seed": j.seed,
            "steps": 8, "cfg": 1.0, "sampler_name": "euler", "scheduler": "beta", "positive": ["pos2", 0],
            "negative": ["neg2", 0], "denoise": j.face_denoise, "feather": 8, "noise_mask": true,
            "force_inpaint": true, "wildcard": "", "cycle": 1}},
        "save": {"class_type": "SaveImage", "inputs": {"filename_prefix": j.prefix, "images": ["fix", 0]}},
    });
    if let Some(l) = j.lora {
        g["lora"] = json!({"class_type": "LoraLoaderModelOnly",
                           "inputs": {"model": ["unet", 0], "lora_name": l.name, "strength_model": l.strength}});
        g["sample"]["inputs"]["model"] = json!(["lora", 0]);
        g["fix"]["inputs"]["model"] = json!(["lora", 0]);
    }
    g
}

/// `image` = the name an upload returned. Output: `save` (mp4).
pub fn video(image: &str, v: &Video, seed: u64, prefix: &str) -> Value {
    // Split as in the Comfy-Org template: high-noise steps 0→2 hand their leftover noise to
    // low-noise steps 2→4.
    let sampler = |model: &str, add_noise: &str, start: u32, end: u32, leftover: &str, latent: Value| {
        json!({"class_type": "KSamplerAdvanced", "inputs": {
            "model": [model, 0], "add_noise": add_noise, "noise_seed": seed, "steps": 4, "cfg": 1.0,
            "sampler_name": "euler", "scheduler": "simple", "positive": ["i2v", 0], "negative": ["i2v", 1],
            "latent_image": latent, "start_at_step": start, "end_at_step": end, "return_with_leftover_noise": leftover}})
    };
    let (w, h) = (v.width, v.height);
    json!({
        "unet_hi": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "unet_lo": {"class_type": "UNETLoader", "inputs": {"unet_name": "wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors", "weight_dtype": "default"}},
        "lora_hi": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_hi", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors", "strength_model": 1.0}},
        "lora_lo": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["unet_lo", 0], "lora_name": "wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors", "strength_model": 1.0}},
        "shift_hi": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_hi", 0], "shift": 8.0}},
        "shift_lo": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["lora_lo", 0], "shift": 8.0}},
        "clip": {"class_type": "CLIPLoader", "inputs": {"clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "default"}},
        "vae": {"class_type": "VAELoader", "inputs": {"vae_name": "wan_2.1_vae.safetensors"}},
        "clipvision": {"class_type": "CLIPVisionLoader", "inputs": {"clip_name": "clip_vision_h.safetensors"}},
        "image": {"class_type": "LoadImage", "inputs": {"image": image}},
        "resize": {"class_type": "ImageScale", "inputs": {"image": ["image", 0], "upscale_method": "lanczos", "width": w, "height": h, "crop": "center"}},
        "cv": {"class_type": "CLIPVisionEncode", "inputs": {"clip_vision": ["clipvision", 0], "image": ["resize", 0], "crop": "none"}},
        "pos": {"class_type": "CLIPTextEncode", "inputs": {"text": v.motion, "clip": ["clip", 0]}},
        "neg": {"class_type": "CLIPTextEncode", "inputs": {"text": VIDEO_NEGATIVE, "clip": ["clip", 0]}},
        "i2v": {"class_type": "WanImageToVideo", "inputs": {"positive": ["pos", 0], "negative": ["neg", 0], "vae": ["vae", 0],
                 "width": w, "height": h, "length": v.length, "batch_size": 1,
                 "clip_vision_output": ["cv", 0], "start_image": ["resize", 0]}},
        "hi": sampler("shift_hi", "enable", 0, 2, "enable", json!(["i2v", 2])),
        "lo": sampler("shift_lo", "disable", 2, 4, "disable", json!(["hi", 0])),
        "decode": {"class_type": "VAEDecode", "inputs": {"samples": ["lo", 0], "vae": ["vae", 0]}},
        "clip_video": {"class_type": "CreateVideo", "inputs": {"images": ["decode", 0], "fps": FPS}},
        "save": {"class_type": "SaveVideo", "inputs": {"video": ["clip_video", 0], "filename_prefix": prefix, "format": "mp4", "codec": "h264"}},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_refs_resolve(g: &Value) {
        let nodes = g.as_object().unwrap();
        for (name, node) in nodes {
            for v in node["inputs"].as_object().unwrap().values() {
                if let Some([Value::String(id), Value::Number(_)]) = v.as_array().map(Vec::as_slice) {
                    assert!(nodes.contains_key(id), "{name} points at missing node {id}");
                }
            }
        }
    }

    fn job(lora: Option<&Lora>) -> Value {
        image(&ImageJob {
            pass1: "p1", pass2: "p2", seed: 7, width: 1152, height: 1728, face_denoise: 0.5, lora, prefix: "x",
        })
    }

    #[test]
    fn face_fix_redraws_the_largest_face_with_pass_two() {
        let g = job(None);
        assert_refs_resolve(&g);
        assert_eq!(g["sample"]["inputs"]["positive"], json!(["pos1", 0]));
        assert_eq!(g["pos1"]["inputs"]["text"], "p1");
        assert_eq!(g["fix"]["inputs"]["positive"], json!(["pos2", 0]));
        assert_eq!(g["pos2"]["inputs"]["text"], "p2");
        assert_eq!(g["fix"]["inputs"]["image"], json!(["decode", 0]));
        assert_eq!(g["fix"]["inputs"]["segs"], json!(["largest", 0]));
        assert_eq!(g["largest"]["inputs"]["take_count"], 1);
        assert_eq!(g["save"]["inputs"]["images"], json!(["fix", 0]));
        assert_eq!(g["before"]["inputs"]["images"], json!(["decode", 0]));
        assert!(g.get("lora").is_none());
    }

    #[test]
    fn lora_feeds_both_passes() {
        let l = Lora { name: "me.safetensors".into(), strength: 0.6 };
        let g = job(Some(&l));
        assert_refs_resolve(&g);
        assert_eq!(g["sample"]["inputs"]["model"], json!(["lora", 0]));
        assert_eq!(g["fix"]["inputs"]["model"], json!(["lora", 0]));
    }

    #[test]
    fn video_starts_from_the_upload() {
        let v = Video { motion: "blinks".into(), length: 49, width: 720, height: 1088 };
        let g = video("still.png", &v, 3, "x");
        assert_refs_resolve(&g);
        assert_eq!(g["image"]["inputs"]["image"], "still.png");
        assert_eq!(g["i2v"]["inputs"]["length"], 49);
        assert_eq!(g["pos"]["inputs"]["text"], "blinks");
        assert_eq!(g["lo"]["inputs"]["latent_image"], json!(["hi", 0]));
    }
}
