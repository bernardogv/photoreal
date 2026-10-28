//! Minimal ComfyUI HTTP client: queue an API-format graph, poll until it finishes,
//! download what the save nodes wrote, upload an input image.

use serde_json::{Value, json};
use std::time::{Duration, Instant};

pub struct Comfy {
    pub base: String,
}

/// A 720×1088 video clip can take several minutes on a single GPU.
const RENDER_TIMEOUT: Duration = Duration::from_secs(1200);
const POLL: Duration = Duration::from_secs(2);

impl Comfy {
    /// POST /prompt, poll /history until done, return the `outputs` map.
    pub fn queue_and_wait(&self, graph: Value) -> Result<Value, String> {
        let queued: Value = ureq::post(format!("{}/prompt", self.base))
            .send_json(json!({"prompt": graph}))
            .map_err(|e| format!("queue: {e}"))?
            .body_mut()
            .read_json()
            .map_err(|e| format!("queue response: {e}"))?;
        let id = queued["prompt_id"]
            .as_str()
            .ok_or_else(|| format!("queue rejected: {queued}"))?
            .to_string();

        let start = Instant::now();
        loop {
            if start.elapsed() > RENDER_TIMEOUT {
                return Err(format!("{id}: no result after {RENDER_TIMEOUT:?}"));
            }
            std::thread::sleep(POLL);
            let hist: Value = ureq::get(format!("{}/history/{id}", self.base))
                .call()
                .map_err(|e| format!("history: {e}"))?
                .body_mut()
                .read_json()
                .map_err(|e| format!("history response: {e}"))?;
            let Some(entry) = hist.get(&id) else { continue };
            let status = &entry["status"];
            if status["completed"].as_bool() != Some(true) {
                if status["status_str"].as_str() == Some("error") {
                    return Err(format!("{id}: execution error: {status}"));
                }
                continue;
            }
            return Ok(entry["outputs"].clone());
        }
    }

    /// Downloads the first file a save node wrote.
    pub fn first_output(&self, outputs: &Value, node: &str) -> Result<Vec<u8>, String> {
        // SaveImage and SaveVideo both report their files under `images`.
        let file = outputs[node]["images"]
            .get(0)
            .ok_or_else(|| format!("node {node} wrote nothing: {outputs}"))?;
        let (Some(name), Some(sub), Some(ty)) = (
            file["filename"].as_str(),
            file["subfolder"].as_str(),
            file["type"].as_str(),
        ) else {
            return Err(format!("malformed output file: {file}"));
        };
        ureq::get(format!("{}/view", self.base))
            .query("filename", name)
            .query("subfolder", sub)
            .query("type", ty)
            .call()
            .map_err(|e| format!("view: {e}"))?
            .body_mut()
            .with_config()
            .limit(512 * 1024 * 1024)
            .read_to_vec()
            .map_err(|e| format!("view body: {e}"))
    }

    /// Uploads a PNG to ComfyUI's input folder; returns the name `LoadImage` wants.
    pub fn upload(&self, name: &str, png: &[u8]) -> Result<String, String> {
        const B: &str = "----photoreal-boundary";
        let mut body = format!(
            "--{B}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{name}\"\r\nContent-Type: image/png\r\n\r\n"
        )
        .into_bytes();
        body.extend_from_slice(png);
        body.extend_from_slice(
            format!("\r\n--{B}\r\nContent-Disposition: form-data; name=\"overwrite\"\r\n\r\ntrue\r\n--{B}--\r\n").as_bytes(),
        );
        let resp: Value = ureq::post(format!("{}/upload/image", self.base))
            .content_type(format!("multipart/form-data; boundary={B}"))
            .send(&body[..])
            .map_err(|e| format!("upload: {e}"))?
            .body_mut()
            .read_json()
            .map_err(|e| format!("upload response: {e}"))?;
        resp["name"]
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| format!("upload rejected: {resp}"))
    }
}
