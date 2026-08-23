import os
import sys
import glob
import time
import random
import ctypes
import subprocess
import gc

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if hasattr(os, "add_dll_directory"):
    for p in [r"C:\mingw64\bin", current_dir, os.path.join(current_dir, "build"), os.getcwd(), os.path.join(os.getcwd(), "build")]:
        if os.path.isdir(p):
            try:
                os.add_dll_directory(p)
            except Exception:
                pass

for bdir in [os.path.join(current_dir, "build"), os.path.join(os.getcwd(), "build"), "/content/Litetorch-/Litetorch-/build", "/content/Litetorch-/build"]:
    for ext in ["liblitetorch.so", "liblitetorch.dll"]:
        lib_file = os.path.join(bdir, ext)
        if os.path.exists(lib_file):
            try:
                ctypes.CDLL(lib_file, mode=ctypes.RTLD_GLOBAL)
                break
            except Exception:
                pass

try:
    from PIL import Image
except ImportError:
    Image = None

import litetorch as lt

def find_dataset_dir():
    for c in ["/content/dataset/data", "../dataset/data", "dataset/data", "dataset", "/content/dataset"]:
        if os.path.isdir(os.path.join(c, "Cat")) and os.path.isdir(os.path.join(c, "Dog")):
            return c
    return "/content/dataset/data"

DATA_DIR = find_dataset_dir()
CAT_DIR = os.path.join(DATA_DIR, "Cat")
DOG_DIR = os.path.join(DATA_DIR, "Dog")
TEST_CAT_IMAGE = os.path.join(DATA_DIR, "cat_test.jpg")
TEST_DOG_IMAGE = os.path.join(DATA_DIR, "dog_test2.jpg")
MODEL_SAVE_PATH = "cat_dog_cnn_eager_model.lt"

IMAGE_SIZE = 64
CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 32
EPOCHS = 25
LEARNING_RATE = 0.0005

def get_ram_usage_mb():
    try:
        with open("/proc/self/status", "r") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    try:
        import resource
        return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
    except Exception:
        return 0.0

def get_gpu_metrics():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        if len(parts) >= 3:
            return float(parts[0]), float(parts[1]), float(parts[2])
    except Exception:
        pass
    return None, None, None

class ConvNetClassifierEager(lt.nn.Module):
    def __init__(self, num_classes=2):
        super().__init__()
        self.training = True
        self.conv1 = lt.nn.Conv2d(3, 32, 3, 1, 1, True)
        self.pool1 = lt.nn.MaxPool2d(2, 2, 0)
        self.conv2 = lt.nn.Conv2d(32, 64, 3, 1, 1, True)
        self.pool2 = lt.nn.MaxPool2d(2, 2, 0)
        self.conv3 = lt.nn.Conv2d(64, 128, 3, 1, 1, True)
        self.pool3 = lt.nn.MaxPool2d(2, 2, 0)
        self.pool4 = lt.nn.MaxPool2d(2, 2, 0)
        self.dropout = lt.nn.Dropout(0.3)
        self.fc1 = lt.nn.Linear(128 * 4 * 4, 128, True)
        self.fc2 = lt.nn.Linear(128, num_classes, True)

    def forward(self, x):
        b = x.shape[0]
        h = self.conv1.forward(x)
        h = lt.Ops.relu(h)
        h = self.pool1.forward(h)

        h = self.conv2.forward(h)
        h = lt.Ops.relu(h)
        h = self.pool2.forward(h)

        h = self.conv3.forward(h)
        h = lt.Ops.relu(h)
        h = self.pool3.forward(h)
        h = self.pool4.forward(h)

        h = self.dropout.forward(h)
        h_flat = h.reshape([b, 128 * 4 * 4]) if hasattr(h, "reshape") else (h.contiguous().view([b, 128 * 4 * 4]) if not h.is_contiguous() else h.view([b, 128 * 4 * 4]))
        h_fc = self.fc1.forward(h_flat)
        h_fc = lt.Ops.relu(h_fc)
        out = self.fc2.forward(h_fc)
        return out

    def train(self, mode=True):
        self.training = mode
        if mode:
            if hasattr(self.dropout, "train"):
                self.dropout.train()
        else:
            if hasattr(self.dropout, "eval"):
                self.dropout.eval()
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        return (
            self.conv1.parameters() +
            self.conv2.parameters() +
            self.conv3.parameters() +
            self.fc1.parameters() +
            self.fc2.parameters()
        )

    def to(self, device):
        self.conv1.to(device)
        self.conv2.to(device)
        self.conv3.to(device)
        self.fc1.to(device)
        self.fc2.to(device)

def load_and_preprocess_image(img_path, size=IMAGE_SIZE, augment=False):
    if Image is None:
        return None
    try:
        with Image.open(img_path) as img:
            if img.mode in ("P", "RGBA", "LA"):
                img = img.convert("RGBA").convert("RGB")
            elif img.mode != "RGB":
                img = img.convert("RGB")
            
            if augment:
                if random.random() > 0.5:
                    img = img.transpose(Image.FLIP_LEFT_RIGHT)
                if random.random() > 0.6:
                    angle = random.uniform(-15.0, 15.0)
                    img = img.rotate(angle, resample=Image.BILINEAR)

            img = img.resize((size, size), Image.BILINEAR)
            raw_bytes = img.tobytes()
            
            total_px = size * size
            ch_r = [0.0] * total_px
            ch_g = [0.0] * total_px
            ch_b = [0.0] * total_px

            mean = (0.485, 0.456, 0.406)
            std = (0.229, 0.224, 0.225)

            inv_255 = 1.0 / 255.0
            inv_std_r = 1.0 / std[0]
            inv_std_g = 1.0 / std[1]
            inv_std_b = 1.0 / std[2]

            for i in range(total_px):
                base_idx = i * 3
                r = raw_bytes[base_idx] * inv_255
                g = raw_bytes[base_idx + 1] * inv_255
                b = raw_bytes[base_idx + 2] * inv_255

                ch_r[i] = (r - mean[0]) * inv_std_r
                ch_g[i] = (g - mean[1]) * inv_std_g
                ch_b[i] = (b - mean[2]) * inv_std_b

            return ch_r + ch_g + ch_b
    except Exception:
        return None

def build_gpu_batches(dataset_pairs, batch_size, channels, height, width, device):
    batches = []
    num_samples = len(dataset_pairs)
    sample_len = channels * height * width

    for start_idx in range(0, num_samples, batch_size):
        chunk = dataset_pairs[start_idx : start_idx + batch_size]
        actual_bs = len(chunk)
        
        flat_x = []
        flat_y = []
        for feat, label in chunk:
            flat_x.extend(feat)
            flat_y.append(label)

        if actual_bs < batch_size:
            pad_count = batch_size - actual_bs
            for _ in range(pad_count):
                flat_x.extend([0.0] * sample_len)
                flat_y.append(0.0)

        x_tensor = lt.Tensor.from_vector(flat_x, [batch_size, channels, height, width], device, False)
        y_tensor = lt.Tensor.from_vector(flat_y, [batch_size], device, False)
        batches.append((x_tensor, y_tensor, actual_bs))

    return batches

def main():
    print("=" * 80)
    print("      LITETORCH CONVNET SYSTEM MONITOR (PURE EAGER EXECUTION)")
    print("=" * 80)

    cat_files = glob.glob(os.path.join(CAT_DIR, "*.jpg")) + glob.glob(os.path.join(CAT_DIR, "*.png"))
    dog_files = glob.glob(os.path.join(DOG_DIR, "*.jpg")) + glob.glob(os.path.join(DOG_DIR, "*.png"))

    all_samples = [(f, 0) for f in cat_files] + [(f, 1) for f in dog_files]
    if not all_samples:
        print(f"Error: No images found in '{CAT_DIR}' or '{DOG_DIR}'")
        return

    random.seed(42)
    random.shuffle(all_samples)
    print(f"Total dataset images found: {len(all_samples)}")

    print("Pre-caching dataset images into system RAM...")
    t_cache_start = time.perf_counter()
    cached_dataset = []
    for path, label in all_samples:
        feat = load_and_preprocess_image(path, augment=True)
        if feat is not None:
            cached_dataset.append((feat, float(label)))
    t_cache_end = time.perf_counter()
    print(f"Pre-cached {len(cached_dataset)} images into RAM in {t_cache_end - t_cache_start:.2f}s | RAM: {get_ram_usage_mb():.1f} MB")

    val_split = int(len(cached_dataset) * 0.8)
    train_data = cached_dataset[:val_split]
    val_data = cached_dataset[val_split:]
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    device = lt.auto_device()
    backend_name = lt.get_backend_name() if hasattr(lt, "get_backend_name") else ("CUDA (Native)" if (hasattr(lt, "cuda") and lt.cuda.is_available()) else ("OpenCL" if lt.is_gpu_available() else "CPU"))
    print(f"Active Compute Device: {device} | Backend Driver: {backend_name} | JIT Engine: Pure Eager (Disabled)")

    gpu_util, vram_used, vram_total = get_gpu_metrics()
    if vram_total is not None:
        print(f"GPU Hardware: NVIDIA GPU | Initial VRAM: {vram_used:.1f} MB / {vram_total:.1f} MB")
    print(f"Host System RAM: {get_ram_usage_mb():.1f} MB")

    print(f"Pre-loading entire dataset into GPU VRAM (NCHW: [3, {IMAGE_SIZE}, {IMAGE_SIZE}])...")
    train_gpu_batches = build_gpu_batches(train_data, BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device)
    val_gpu_batches = build_gpu_batches(val_data, BATCH_SIZE, CHANNELS, IMAGE_SIZE, IMAGE_SIZE, device)
    gpu_util, vram_used_after, _ = get_gpu_metrics()
    if vram_used_after is not None and vram_used is not None:
        print(f"Dataset VRAM Footprint: {vram_used_after - vram_used:.2f} MB | Current VRAM: {vram_used_after:.1f} MB")
    print("================================================================================\n")

    random.seed(42)
    model = ConvNetClassifierEager(NUM_CLASSES)
    model.to(device)
    optimizer = lt.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_train_batches = len(train_gpu_batches)
    print("Beginning Pure Eager ConvNet Model Training Pipeline:")
    total_training_start = time.perf_counter()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        epoch_cpu_start = time.process_time()

        model.train()
        random.shuffle(train_gpu_batches)
        correct = 0
        total = 0
        epoch_gpu_compute_time = 0.0

        for batch_idx, (x_tensor, y_tensor, actual_batch_size) in enumerate(train_gpu_batches, start=1):
            t_gpu_start = time.perf_counter()
            optimizer.zero_grad()
            logits = model.forward(x_tensor)
            loss = lt.Ops.cross_entropy_loss(logits, y_tensor)
            loss.backward()
            optimizer.step()
            t_gpu_end = time.perf_counter()

            gpu_step_time = (t_gpu_end - t_gpu_start)
            epoch_gpu_compute_time += gpu_step_time
            total += actual_batch_size

            ram_mb = get_ram_usage_mb()
            g_util, v_used, v_tot = get_gpu_metrics()

            vram_str = f"VRAM: {v_used:.0f}/{v_tot:.0f}MB" if v_used is not None else "VRAM: N/A"
            gpu_str = f"GPU: {g_util:.0f}%" if g_util is not None else ""

            bar_len = 15
            filled_len = int(bar_len * batch_idx // total_train_batches)
            bar = "=" * filled_len + ">" + "." * (bar_len - filled_len - 1) if filled_len < bar_len else "=" * bar_len

            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{EPOCHS}] [{bar}] {batch_idx}/{total_train_batches} | "
                f"GPU Time: {gpu_step_time*1000.0:4.1f}ms | RAM: {ram_mb:.0f}MB | {vram_str} {gpu_str}"
            )
            sys.stdout.flush()

        avg_loss = loss.item()

        model.eval()
        val_correct = 0
        val_total = 0

        for x_tensor, y_tensor, actual_batch_size in val_gpu_batches:
            logits = model.forward(x_tensor)
            logits_vec = logits.to_vector()
            y_vec = y_tensor.to_vector()
            for b in range(actual_batch_size):
                pred = 0 if logits_vec[b * 2] > logits_vec[b * 2 + 1] else 1
                if pred == int(y_vec[b]):
                    val_correct += 1
                val_total += 1

        val_acc = (val_correct / val_total * 100.0) if val_total > 0 else 0.0

        saved_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            lt.save(model, MODEL_SAVE_PATH)
            saved_marker = " *"

        lt.empty_cache()
        gc.collect()

        epoch_end_time = time.perf_counter()
        epoch_cpu_end = time.process_time()

        epoch_wall_time = epoch_end_time - epoch_start_time
        epoch_cpu_time = epoch_cpu_end - epoch_cpu_start

        sys.stdout.write("\r" + " " * 120 + "\r")
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Val Acc: {val_acc:5.2f}%{saved_marker} | "
            f"GPU Pure Compute: {epoch_gpu_compute_time:5.2f}s | CPU Time: {epoch_cpu_time:5.2f}s | Wall Time: {epoch_wall_time:5.2f}s | RAM: {get_ram_usage_mb():.1f}MB"
        )

    total_training_end = time.perf_counter()
    print("================================================================================")
    print(f"Training Complete! Total Wall Time: {total_training_end - total_training_start:.2f}s | Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})")

    if os.path.exists(MODEL_SAVE_PATH):
        lt.load(model, MODEL_SAVE_PATH)
        file_size_kb = os.path.getsize(MODEL_SAVE_PATH) / 1024.0
        print(f"Loaded Best Model from '{MODEL_SAVE_PATH}' ({file_size_kb:.2f} KB) for Inference Benchmark.")
    print("================================================================================\n")

    print("Running Inference & Latency Benchmark on Test Images:")
    model.eval()
    test_images = [TEST_CAT_IMAGE, TEST_DOG_IMAGE]
    classes = ["Cat", "Dog"]

    for img_path in test_images:
        if os.path.exists(img_path):
            t_load = time.perf_counter()
            feat = load_and_preprocess_image(img_path, augment=False)
            t_load_done = time.perf_counter()

            if feat is not None:
                x_tensor = lt.Tensor.from_vector(feat, [1, CHANNELS, IMAGE_SIZE, IMAGE_SIZE], device, False)
                t_infer_start = time.perf_counter()
                logits = model.forward(x_tensor)
                probs = lt.Ops.softmax(logits, -1).to_vector()
                t_infer_end = time.perf_counter()

                pred_class = classes[0] if probs[0] > probs[1] else classes[1]
                conf = max(probs[0], probs[1]) * 100.0
                infer_latency_ms = (t_infer_end - t_infer_start) * 1000.0
                load_latency_ms = (t_load_done - t_load) * 1000.0

                print(
                    f"Image: {img_path:30s} -> Predicted: {pred_class:3s} ({conf:5.2f}% confidence) | "
                    f"Inference Latency: {infer_latency_ms:5.2f}ms | Load Latency: {load_latency_ms:5.2f}ms"
                )
        else:
            print(f"Test image '{img_path}' not found.")

if __name__ == "__main__":
    main()
