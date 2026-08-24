import os
import sys
import glob
import time
import random
import queue
import threading
import subprocess
from concurrent.futures import ThreadPoolExecutor
try:
    from PIL import Image
except ImportError:
    Image = None

for dll_dir in [r"C:\mingw64\bin", os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()]:
    if os.path.isdir(dll_dir) and hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(dll_dir)
        except Exception:
            pass

try:
    import litetorch as lt
except ImportError:
    current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
    for search_path in [current_dir, os.path.join(current_dir, "build"), os.getcwd(), os.path.join(os.getcwd(), "build")]:
        if search_path not in sys.path and os.path.isdir(search_path):
            sys.path.insert(0, search_path)
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
MODEL_SAVE_PATH = "cat_dog_cnn_model.lt"

IMAGE_SIZE = 64
CHANNELS = 3
NUM_CLASSES = 2
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.0005
NUM_WORKERS = 4
PREFETCH_BATCHES = 4

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

class ConvNetClassifier(lt.nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
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

        v_x1 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x1")
        self.jit_relu1 = lt.jit.JITFunction("fused_conv1_relu", lt.jit.relu(v_x1), [v_x1])
        v_x2 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x2")
        self.jit_relu2 = lt.jit.JITFunction("fused_conv2_relu", lt.jit.relu(v_x2), [v_x2])
        v_x3 = lt.jit.JITVar(lt.jit.OpType.INPUT, "x3")
        self.jit_relu3 = lt.jit.JITFunction("fused_conv3_relu", lt.jit.relu(v_x3), [v_x3])
        v_xfc = lt.jit.JITVar(lt.jit.OpType.INPUT, "xfc")
        self.jit_relu_fc = lt.jit.JITFunction("fused_fc_relu", lt.jit.relu(v_xfc), [v_xfc])

    def forward(self, x):
        b = x.shape[0]
        h = self.conv1.forward(x)
        h = self.jit_relu1([h])
        h = self.pool1.forward(h)

        h = self.conv2.forward(h)
        h = self.jit_relu2([h])
        h = self.pool2.forward(h)

        h = self.conv3.forward(h)
        h = self.jit_relu3([h])
        h = self.pool3.forward(h)
        h = self.pool4.forward(h)

        h = self.dropout.forward(h)
        h_flat = h.reshape([b, 128 * 4 * 4]) if hasattr(h, "reshape") else (h.contiguous().view([b, 128 * 4 * 4]) if not h.is_contiguous() else h.view([b, 128 * 4 * 4]))
        h_fc = self.fc1.forward(h_flat)
        h_fc = self.jit_relu_fc([h_fc])
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
    try:
        with Image.open(img_path) as img:
            if img.mode in ("P", "RGBA", "LA"):
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
            if augment and random.random() > 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
            img = img.resize((size, size))
            w, h = img.size
            r_c, g_c, b_c = [], [], []
            for y in range(h):
                for x in range(w):
                    r, g, b = img.getpixel((x, y))
                    r_c.append((r / 255.0 - 0.5) * 2.0)
                    g_c.append((g / 255.0 - 0.5) * 2.0)
                    b_c.append((b / 255.0 - 0.5) * 2.0)
            return r_c + g_c + b_c
    except Exception:
        return None

def process_single_task(args):
    img_path, label, size, augment = args
    feat = load_and_preprocess_image(img_path, size, augment)
    return (feat, label) if feat is not None else None

class StreamingDataLoader:
    def __init__(self, samples, batch_size, channels, size, device, shuffle=True, augment=True, num_workers=NUM_WORKERS, prefetch_batches=PREFETCH_BATCHES):
        self.samples = list(samples)
        self.batch_size = batch_size
        self.channels = channels
        self.size = size
        self.device = device
        self.shuffle = shuffle
        self.augment = augment
        self.num_workers = num_workers
        self.prefetch_batches = prefetch_batches
        self.total_samples = len(self.samples)
        self.total_batches = (self.total_samples + batch_size - 1) // batch_size

    def __len__(self):
        return self.total_batches

    def __iter__(self):
        if self.shuffle:
            random.shuffle(self.samples)

        batch_queue = queue.Queue(maxsize=self.prefetch_batches)
        stop_token = object()

        def producer():
            with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                for i in range(0, self.total_samples, self.batch_size):
                    batch_samples = self.samples[i:i + self.batch_size]
                    tasks = [(p, lbl, self.size, self.augment) for p, lbl in batch_samples]
                    results = list(executor.map(process_single_task, tasks))
                    valid_items = [r for r in results if r is not None]
                    if not valid_items:
                        continue
                    actual_b = len(valid_items)
                    batch_x = [v for item in valid_items for v in item[0]]
                    batch_y = [float(item[1]) for item in valid_items]
                    x_tensor = lt.Tensor.from_vector(batch_x, [actual_b, self.channels, self.size, self.size], self.device, False)
                    y_tensor = lt.Tensor.from_vector(batch_y, [actual_b], self.device, False)
                    batch_queue.put((x_tensor, y_tensor, actual_b))
            batch_queue.put(stop_token)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        while True:
            item = batch_queue.get()
            if item is stop_token:
                break
            yield item

def load_dataset():
    samples = []
    cat_files = glob.glob(os.path.join(CAT_DIR, "*.*"))
    for p in cat_files:
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            samples.append((p, 0))
    dog_files = glob.glob(os.path.join(DOG_DIR, "*.*"))
    for p in dog_files:
        if p.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            samples.append((p, 1))
    random.seed(42)
    random.shuffle(samples)
    return samples

def main():
    if not os.path.exists(CAT_DIR) or not os.path.exists(DOG_DIR):
        print(f"Error: Folders '{CAT_DIR}' or '{DOG_DIR}' not found.")
        return

    samples = load_dataset()
    print("================================================================================")
    print("           LITETORCH HIGH-SCALE STREAMING CNN DEEP LEARNING")
    print("================================================================================")
    print(f"Total dataset images found: {len(samples)}")
    if len(samples) == 0:
        print("No images found in dataset/data/Cat or dataset/data/Dog.")
        return

    val_split = int(len(samples) * 0.8)
    train_samples = samples[:val_split]
    val_samples = samples[val_split:]
    print(f"Train samples: {len(train_samples)}, Validation samples: {len(val_samples)}")

    device = lt.auto_device()
    backend_name = lt.get_backend_name() if hasattr(lt, "get_backend_name") else ("CUDA (Native)" if (hasattr(lt, "cuda") and lt.cuda.is_available()) else ("OpenCL" if lt.is_gpu_available() else "CPU"))
    jit_status = "Enabled (Kernel Fusion & JIT Autograd)" if (hasattr(lt, "jit") and hasattr(lt.jit, "JITFunction")) else "Disabled"
    print(f"Active Compute Device: {device} | Backend Driver: {backend_name} | JIT Engine: {jit_status}")
    
    gpu_util, vram_used, vram_total = get_gpu_metrics()
    if vram_total is not None:
        print(f"GPU Hardware: NVIDIA GPU | Initial VRAM: {vram_used:.1f} MB / {vram_total:.1f} MB")
    print(f"Host System RAM: {get_ram_usage_mb():.1f} MB")
    print(f"Model Configuration: Scaled ConvNet-3L | Batch Size: {BATCH_SIZE}")
    print("Data Pipeline: Streaming Multi-Threaded Prefetching (Constant RAM & VRAM Footprint)")
    print("================================================================================\n")

    train_loader = StreamingDataLoader(train_samples, BATCH_SIZE, CHANNELS, IMAGE_SIZE, device, shuffle=True, augment=True)
    val_loader = StreamingDataLoader(val_samples, BATCH_SIZE, CHANNELS, IMAGE_SIZE, device, shuffle=False, augment=False)

    random.seed(42)
    model = ConvNetClassifier(NUM_CLASSES)
    model.to(device)
    optimizer = lt.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_train_batches = len(train_loader)
    print("Beginning Scaled CNN Training Pipeline:")
    total_training_start = time.perf_counter()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        epoch_cpu_start = time.process_time()

        model.train()
        total_loss = 0.0
        total_samples_processed = 0
        epoch_gpu_compute_time = 0.0
        last_loss = 0.0

        for batch_idx, (x_tensor, y_tensor, actual_batch_size) in enumerate(train_loader, start=1):
            t_gpu_start = time.perf_counter()
            optimizer.zero_grad()
            logits = model.forward(x_tensor)
            loss = lt.Ops.cross_entropy_loss(logits, y_tensor)
            loss.backward()
            optimizer.step()
            t_gpu_end = time.perf_counter()

            gpu_step_time = (t_gpu_end - t_gpu_start)
            epoch_gpu_compute_time += gpu_step_time
            total_samples_processed += actual_batch_size
            last_loss = loss.item()
            total_loss += last_loss * actual_batch_size

            ram_mb = get_ram_usage_mb()
            g_util, v_used, v_tot = get_gpu_metrics()

            vram_str = f"VRAM: {v_used:.0f}/{v_tot:.0f}MB" if v_used is not None else "VRAM: N/A"
            gpu_str = f"GPU: {g_util:.0f}%" if g_util is not None else ""

            bar_len = 15
            filled_len = int(bar_len * batch_idx // total_train_batches)
            bar = "=" * filled_len + ">" + "." * (bar_len - filled_len - 1) if filled_len < bar_len else "=" * bar_len

            throughput = total_samples_processed / (time.perf_counter() - epoch_start_time + 1e-6)
            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{EPOCHS}] [{bar}] {batch_idx}/{total_train_batches} | "
                f"{throughput:5.1f} img/s | GPU: {gpu_step_time*1000.0:4.1f}ms | RAM: {ram_mb:.0f}MB | {vram_str} {gpu_str}"
            )
            sys.stdout.flush()

        avg_loss = (total_loss / total_samples_processed) if total_samples_processed > 0 else last_loss

        model.eval()
        val_correct = 0
        val_total = 0

        for x_tensor, y_tensor, actual_batch_size in val_loader:
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

        if hasattr(lt, "empty_cache"):
            lt.empty_cache()

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
