"""Small model constants shared by local PyTorch and serverless ONNX runtimes."""

CLASS_LABELS = (
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
)
CREMA_D_CLASS_LABELS = (
    "neutral",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
)
MODEL_ARCHITECTURES = ("cnn", "cnn_lstm", "cnn_bilstm_attention")
