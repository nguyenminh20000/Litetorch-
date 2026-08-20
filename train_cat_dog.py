import os
import sys
import glob
import time
import random
import ctypes
import subprocess

current_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in locals() else os.getcwd()
if hasattr(os, "add_dll_directory"):
    for p in [current_dir, os.path.join(current_dir, "build"), os.getcwd(), os.path.join(os.getcwd(), "build")]:
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

from PIL import Image
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
MODEL_SAVE_PATH = "cat_dog_model.lt"

IMAGE_SIZE = 64
CHANNELS = 3
PATCH_SIZE = 16
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) * (IMAGE_SIZE // PATCH_SIZE)
PATCH_DIM = PATCH_SIZE * PATCH_SIZE * CHANNELS
EMBED_DIM = 128
NUM_HEADS = 4
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

class TransformerBlock(lt.nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim):
        super().__init__()
        self.attn = lt.nn.MultiHeadAttention(embed_dim, num_heads)
        self.ln1 = lt.nn.LayerNorm([embed_dim])
        self.fc1 = lt.nn.Linear(embed_dim, mlp_dim, True)
        self.fc2 = lt.nn.Linear(mlp_dim, embed_dim, True)
        self.ln2 = lt.nn.LayerNorm([embed_dim])

    def forward(self, x):
        norm1 = self.ln1.forward(x)
        attn_out = self.attn.forward(norm1)
        x = lt.Ops.add(x, attn_out)
        norm2 = self.ln2.forward(x)
        mlp_h = self.fc1.forward(norm2)
        mlp_h = lt.Ops.relu(mlp_h)
        mlp_out = self.fc2.forward(mlp_h)
        x = lt.Ops.add(x, mlp_out)
        return x

    def parameters(self):
        return self.attn.parameters() + self.ln1.parameters() + self.fc1.parameters() + self.fc2.parameters() + self.ln2.parameters()

    def to(self, device):
        self.attn.to(device)
        self.ln1.to(device)
        self.fc1.to(device)
        self.fc2.to(device)
        self.ln2.to(device)

class VisionTransformerClassifier(lt.nn.Module):
    def __init__(self, num_patches, patch_dim, embed_dim, num_heads, num_classes):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.patch_proj = lt.nn.Linear(patch_dim, embed_dim, True)
        self.block1 = TransformerBlock(embed_dim, num_heads, embed_dim * 2)
        self.block2 = TransformerBlock(embed_dim, num_heads, embed_dim * 2)
        self.ln_f = lt.nn.LayerNorm([embed_dim])
        self.head = lt.nn.Linear(num_patches * embed_dim, num_classes, True)

    def forward(self, x):
        b = x.shape[0]
        h = self.patch_proj.forward(x)
        h = self.block1.forward(h)
        h = self.block2.forward(h)
        h = self.ln_f.forward(h)
        h_flat = h.view([b, self.num_patches * self.embed_dim])
        out = self.head.forward(h_flat)
        return out

    def parameters(self):
        return self.patch_proj.parameters() + self.block1.parameters() + self.block2.parameters() + self.ln_f.parameters() + self.head.parameters()

    def to(self, device):
        self.patch_proj.to(device)
        self.block1.to(device)
        self.block2.to(device)
        self.ln_f.to(device)
        self.head.to(device)

def load_and_preprocess_image(img_path, size=IMAGE_SIZE, patch_size=PATCH_SIZE):
    try:
        with Image.open(img_path) as img:
            if img.mode in ("P", "RGBA", "LA"):
                img = img.convert("RGBA").convert("RGB")
            else:
                img = img.convert("RGB")
            img = img.resize((size, size))
            w, h = img.size
            patches = []
            for r in range(0, h, patch_size):
                for c in range(0, w, patch_size):
                    patch = img.crop((c, r, c + patch_size, r + patch_size))
                    raw_bytes = patch.tobytes()
                    patches.extend([b / 255.0 for b in raw_bytes])
            return patches
    except Exception:
        return None

def load_dataset():
    samples = []
    
    cat_files = glob.glob(os.path.join(CAT_DIR, "*.*"))
    for p in cat_files:
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
            samples.append((p, 0))
            
    dog_files = glob.glob(os.path.join(DOG_DIR, "*.*"))
    for p in dog_files:
        if p.lower().endswith((".jpg", ".jpeg", ".png")):
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
    print("                 LITETORCH DEEP LEARNING SYSTEM MONITOR")
    print("================================================================================")
    print(f"Total dataset images found: {len(samples)}")
    if len(samples) == 0:
        print("No images found in dataset/data/Cat or dataset/data/Dog.")
        return

    print("Pre-caching dataset images into system RAM...")
    t_cache_start = time.perf_counter()
    cached_dataset = []
    for path, label in samples:
        feat = load_and_preprocess_image(path)
        if feat is not None:
            cached_dataset.append((feat, float(label)))
    t_cache_end = time.perf_counter()
    print(f"Pre-cached {len(cached_dataset)} images into RAM in {t_cache_end - t_cache_start:.2f}s | RAM: {get_ram_usage_mb():.1f} MB")

    val_split = int(len(cached_dataset) * 0.8)
    train_data = cached_dataset[:val_split]
    val_data = cached_dataset[val_split:]
    print(f"Train samples: {len(train_data)}, Validation samples: {len(val_data)}")

    device = lt.auto_device()
    print(f"Active Compute Device: {device}")
    
    gpu_util, vram_used, vram_total = get_gpu_metrics()
    if vram_total is not None:
        print(f"GPU Hardware: NVIDIA GPU | Initial VRAM: {vram_used:.1f} MB / {vram_total:.1f} MB")
    print(f"Host System RAM: {get_ram_usage_mb():.1f} MB")
    print("================================================================================\n")

    random.seed(42)
    model = VisionTransformerClassifier(NUM_PATCHES, PATCH_DIM, EMBED_DIM, NUM_HEADS, NUM_CLASSES)
    model.to(device)
    optimizer = lt.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    total_train_batches = (len(train_data) + BATCH_SIZE - 1) // BATCH_SIZE

    print("Beginning Model Training Pipeline:")
    total_training_start = time.perf_counter()
    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(1, EPOCHS + 1):
        epoch_start_time = time.perf_counter()
        epoch_cpu_start = time.process_time()

        random.shuffle(train_data)
        total_loss = 0.0
        correct = 0
        total = 0

        epoch_gpu_compute_time = 0.0

        for batch_idx, i in enumerate(range(0, len(train_data), BATCH_SIZE), start=1):
            batch = train_data[i:i + BATCH_SIZE]
            actual_batch_size = len(batch)
            if actual_batch_size == 0:
                continue

            batch_x = [v for item in batch for v in item[0]]
            batch_y = [item[1] for item in batch]

            t_gpu_start = time.perf_counter()
            x_tensor = lt.Tensor.from_vector(batch_x, [actual_batch_size, NUM_PATCHES, PATCH_DIM], device, False)
            y_tensor = lt.Tensor.from_vector(batch_y, [actual_batch_size], device, False)

            optimizer.zero_grad()
            logits = model.forward(x_tensor)
            loss = lt.Ops.cross_entropy_loss(logits, y_tensor)
            loss.backward()
            optimizer.step()
            t_gpu_end = time.perf_counter()

            gpu_step_time = (t_gpu_end - t_gpu_start)
            epoch_gpu_compute_time += gpu_step_time

            total_loss += loss.item() * actual_batch_size

            logits_vec = logits.to_vector()
            for b in range(actual_batch_size):
                pred = 0 if logits_vec[b * 2] > logits_vec[b * 2 + 1] else 1
                if pred == int(batch_y[b]):
                    correct += 1
                total += 1

            current_train_acc = (correct / total * 100.0) if total > 0 else 0.0
            current_loss = loss.item()
            ram_mb = get_ram_usage_mb()
            g_util, v_used, v_tot = get_gpu_metrics()

            vram_str = f"VRAM: {v_used:.0f}/{v_tot:.0f}MB" if v_used is not None else "VRAM: N/A"
            gpu_str = f"GPU: {g_util:.0f}%" if g_util is not None else ""

            bar_len = 15
            filled_len = int(bar_len * batch_idx // total_train_batches)
            bar = "=" * filled_len + ">" + "." * (bar_len - filled_len - 1) if filled_len < bar_len else "=" * bar_len

            sys.stdout.write(
                f"\rEpoch [{epoch:2d}/{EPOCHS}] [{bar}] {batch_idx}/{total_train_batches} | "
                f"Loss: {current_loss:.4f} | Acc: {current_train_acc:.1f}% | "
                f"GPU Time: {gpu_step_time*1000.0:4.1f}ms | RAM: {ram_mb:.0f}MB | {vram_str} {gpu_str}"
            )
            sys.stdout.flush()

        avg_loss = total_loss / (total if total > 0 else 1)
        train_acc = (correct / total * 100.0) if total > 0 else 0.0

        val_correct = 0
        val_total = 0

        for i in range(0, len(val_data), BATCH_SIZE):
            batch = val_data[i:i + BATCH_SIZE]
            actual_batch_size = len(batch)
            if actual_batch_size == 0:
                continue

            batch_x = [v for item in batch for v in item[0]]
            batch_y = [item[1] for item in batch]

            x_tensor = lt.Tensor.from_vector(batch_x, [actual_batch_size, NUM_PATCHES, PATCH_DIM], device, False)
            logits = model.forward(x_tensor)
            logits_vec = logits.to_vector()
            for b in range(actual_batch_size):
                pred = 0 if logits_vec[b * 2] > logits_vec[b * 2 + 1] else 1
                if pred == int(batch_y[b]):
                    val_correct += 1
                val_total += 1

        val_acc = (val_correct / val_total * 100.0) if val_total > 0 else 0.0

        saved_marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            lt.save(model, MODEL_SAVE_PATH)
            saved_marker = " *"

        epoch_end_time = time.perf_counter()
        epoch_cpu_end = time.process_time()

        epoch_wall_time = epoch_end_time - epoch_start_time
        epoch_cpu_time = epoch_cpu_end - epoch_cpu_start

        sys.stdout.write("\r" + " " * 120 + "\r")
        print(
            f"Epoch {epoch:2d}/{EPOCHS} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:5.2f}% | Val Acc: {val_acc:5.2f}%{saved_marker} | "
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
    test_images = [TEST_CAT_IMAGE, TEST_DOG_IMAGE]
    classes = ["Cat", "Dog"]

    for img_path in test_images:
        if os.path.exists(img_path):
            t_load = time.perf_counter()
            feat = load_and_preprocess_image(img_path)
            t_load_done = time.perf_counter()

            if feat is not None:
                x_tensor = lt.Tensor.from_vector(feat, [1, NUM_PATCHES, PATCH_DIM], device, False)
                
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
