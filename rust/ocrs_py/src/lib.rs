use std::env;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use anyhow::{anyhow, Context};
use numpy::ndarray::s;
use numpy::PyReadonlyArray3;
use ocrs::{DecodeMethod, DimOrder, ImageSource, OcrEngine, OcrEngineParams};
use ocrs::TextItem;
use once_cell::sync::OnceCell;
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use pyo3::IntoPy;
use rten::Model;
use rten_imageproc::BoundingRect;
use rten_tensor::NdTensor;
use rten_tensor::AsView;

const DETECTION_MODEL_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-detection.rten";
const RECOGNITION_MODEL_URL: &str = "https://ocrs-models.s3-accelerate.amazonaws.com/text-recognition.rten";

struct ModelPaths {
    detection: PathBuf,
    recognition: PathBuf,
}

static MODEL_PATHS: OnceCell<ModelPaths> = OnceCell::new();
static DEFAULT_ENGINE: OnceCell<Mutex<OcrEngine>> = OnceCell::new();

fn cache_dir() -> anyhow::Result<PathBuf> {
    let mut cache_dir = home::home_dir().ok_or_else(|| anyhow!("Failed to determine home directory"))?;
    cache_dir.push(".cache");
    cache_dir.push("ocrs");
    std::fs::create_dir_all(&cache_dir)?;
    Ok(cache_dir)
}

fn filename_from_url(url: &str) -> anyhow::Result<String> {
    let parsed = url::Url::parse(url)?;
    let path = Path::new(parsed.path());
    path.file_name()
        .and_then(|p| p.to_str())
        .map(|s| s.to_string())
        .ok_or_else(|| anyhow!("Could not determine filename for {}", url))
}

fn download_file(url: &str, filename: Option<&str>) -> anyhow::Result<PathBuf> {
    let cache_dir = cache_dir()?;
    let filename = filename
        .map(|f| f.to_string())
        .map(Ok)
        .unwrap_or_else(|| filename_from_url(url))?;

    if let Ok(model_dir) = env::var("OCRS_MODEL_DIR") {
        let local_path = Path::new(&model_dir).join(&filename);
        if local_path.exists() {
            return Ok(local_path);
        }
    }

    let destination = cache_dir.join(filename);
    if destination.exists() {
        return Ok(destination);
    }

    let response = ureq::get(url).call()?;
    let mut reader = response.into_reader();
    let mut buf = Vec::new();
    use std::io::Read;
    reader.read_to_end(&mut buf)?;
    std::fs::write(&destination, &buf)?;

    Ok(destination)
}

fn load_models() -> anyhow::Result<&'static ModelPaths> {
    MODEL_PATHS.get_or_try_init(|| {
        let detection = download_file(DETECTION_MODEL_URL, None)
            .context("Failed to download detection model")?;
        let recognition = download_file(RECOGNITION_MODEL_URL, None)
            .context("Failed to download recognition model")?;
        Ok(ModelPaths {
            detection,
            recognition,
        })
    })
}

fn build_engine(allowed_chars: Option<String>, beam_search: bool) -> anyhow::Result<OcrEngine> {
    let models = load_models()?;
    let detection_path = env::var("OCRS_DETECTION_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models.detection.clone());
    let recognition_path = env::var("OCRS_RECOGNITION_MODEL")
        .map(PathBuf::from)
        .unwrap_or_else(|_| models.recognition.clone());

    let detection_model = Model::load_file(&detection_path)
        .with_context(|| format!("Failed to load detection model from {:?}", detection_path))?;
    let recognition_model = Model::load_file(&recognition_path)
        .with_context(|| format!("Failed to load recognition model from {:?}", recognition_path))?;

    OcrEngine::new(OcrEngineParams {
        detection_model: Some(detection_model),
        recognition_model: Some(recognition_model),
        decode_method: if beam_search {
            DecodeMethod::BeamSearch { width: 100 }
        } else {
            DecodeMethod::Greedy
        },
        allowed_chars,
        ..Default::default()
    })
}

fn run_engine(engine: &OcrEngine, image: PyReadonlyArray3<u8>) -> anyhow::Result<OcrOutput> {
    let view = image.as_array();
    let shape = view.shape();
    if shape.len() != 3 {
        return Err(anyhow!("Expected image with 3 dimensions (H, W, C)"));
    }
    let height = shape[0];
    let width = shape[1];
    let channels = shape[2];

    if channels != 1 && channels != 3 && channels != 4 {
        return Err(anyhow!("Expected image with 1, 3 or 4 channels"));
    }

    let mut owned = view.to_owned();
    let mut channel_count = channels;
    if channels == 4 {
        owned = view.slice(s![.., .., 0..3]).to_owned();
        channel_count = 3;
    }

    let data = owned.into_raw_vec();

    let tensor = NdTensor::from_data([height, width, channel_count], data);
    let source = ImageSource::from_tensor(tensor.view(), DimOrder::Hwc)?;
    let input = engine.prepare_input(source)?;
    let word_rects = engine.detect_words(&input)?;
    let line_rects = engine.find_text_lines(&input, &word_rects);
    let text_lines = engine.recognize_text(&input, &line_rects)?;

    Ok(OcrOutput::from_text_lines(&text_lines))
}

#[derive(Default)]
struct OcrOutput {
    texts: Vec<String>,
    left: Vec<i32>,
    top: Vec<i32>,
    width: Vec<i32>,
    height: Vec<i32>,
    conf: Vec<String>,
}

impl OcrOutput {
    fn from_text_lines(text_lines: &[Option<ocrs::TextLine>]) -> Self {
        let mut output = OcrOutput::default();
        for line in text_lines.iter().flatten() {
            for word in line.words() {
                let text = word.to_string();
                if text.trim().is_empty() {
                    continue;
                }
                let rect = word.rotated_rect().bounding_rect();
                output.texts.push(text);
                output.left.push(rect.left() as i32);
                output.top.push(rect.top() as i32);
                output.width.push(rect.width() as i32);
                output.height.push(rect.height() as i32);
                output.conf.push("100".to_string());
            }
        }
        output
    }
}

#[pyfunction]
fn run_ocr(
    py: Python,
    image: PyReadonlyArray3<u8>,
    allowed_chars: Option<String>,
    beam_search: Option<bool>,
) -> PyResult<PyObject> {
    let beam_search = beam_search.unwrap_or(false);
    let output = if allowed_chars.is_none() && !beam_search {
        let engine_mutex = DEFAULT_ENGINE
            .get_or_try_init(|| {
                build_engine(None, false).map(Mutex::new)
            })
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        let engine = engine_mutex
            .lock()
            .map_err(|_| PyRuntimeError::new_err("Failed to acquire OCR engine"))?;
        run_engine(&engine, image).map_err(|err| PyRuntimeError::new_err(err.to_string()))?
    } else {
        let engine = build_engine(allowed_chars.clone(), beam_search)
            .map_err(|err| PyRuntimeError::new_err(err.to_string()))?;
        run_engine(&engine, image).map_err(|err| PyRuntimeError::new_err(err.to_string()))?
    };

    let dict = PyDict::new_bound(py);
    dict.set_item("text", PyList::new_bound(py, &output.texts))?;
    dict.set_item("left", PyList::new_bound(py, &output.left))?;
    dict.set_item("top", PyList::new_bound(py, &output.top))?;
    dict.set_item("width", PyList::new_bound(py, &output.width))?;
    dict.set_item("height", PyList::new_bound(py, &output.height))?;
    dict.set_item("conf", PyList::new_bound(py, &output.conf))?;
    Ok(dict.into_py(py))
}

#[pymodule]
fn _ocrs(py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(run_ocr, m)?)?;
    // Ensure default engine initializes eagerly so build errors surface early.
    py.allow_threads(|| {
        if let Err(err) = DEFAULT_ENGINE.get_or_try_init(|| build_engine(None, false).map(Mutex::new)) {
            let _ = err; // ignore initialization errors during import
        }
    });
    Ok(())
}
