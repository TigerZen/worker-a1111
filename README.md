<h1>Automatic1111 Stable Diffusion web UI</h1>

[![RunPod](https://api.runpod.io/badge/runpod-workers/worker-a1111)](https://www.runpod.io/console/hub/runpod-workers/worker-a1111)

- Runs [Automatic1111 Stable Diffusion WebUI](https://github.com/AUTOMATIC1111/stable-diffusion-webui) and exposes its `txt2img` API endpoint
- Comes pre-packaged with the [**Deliberate v6**](https://huggingface.co/XpucT/Deliberate) model

---

## Usage

The `input` object accepts any valid parameter for the Automatic1111 `/sdapi/v1/txt2img` endpoint. Refer to the [Automatic1111 API Documentation](https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/API) for a full list of available parameters (like `seed`, `sampler_name`, `batch_size`, `styles`, `override_settings`, etc.).

### Example Request

Here's an example payload to generate an image:

```json
{
  "input": {
    "prompt": "Realistic photograph, Movie light, film grain, score_9, score_8_up, score_7_up, source_real, source_photo, source_realistic, source_realism, vivid colors, depth of field, masterpiece, 4k, high quality, (best quality:1.1), A stylish woman, 1920s flapper style, elegant dress, confident pose, sepia tone, soft lighting, vintage photography style, detailed, 8k resolution, art deco background, ",
    "negative_prompt": "worst quality,low quality,worst detail,low detail,bad anatomy, bad hands,text,error,missing fingers,(extra digit:1.4),fewer digits,cropped,signature,watermark,username,blurry, ",
    "steps": 25,
    "cfg_scale": 7,
    "width": 1024,
    "height": 1024,
    "sampler_name": "DPM++ 2M SDE Karras"
  }
}
```
