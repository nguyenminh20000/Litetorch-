#!/usr/bin/env python3
import litetorch as lt
import time
import os

try:
    import resource
except ImportError:
    resource = None

def current_rss_mb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return float(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0

def peak_rss_mb():
    if resource is not None:
        try:
            return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0
        except Exception:
            pass
    return 0.0

def get_vram_mb():
    try:
        out = os.popen("nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null").read().strip()
        if out:
            return float(out.split('\n')[0])
    except Exception:
        pass
    return 0.0

def generate_spiral_data(samples=600, classes=3):
    import math, random
    random.seed(42)
    x_data = []
    y_data = []
    per_class = samples // classes
    pi = math.pi
    for c in range(classes):
        for i in range(per_class):
            t = float(i) / float(per_class - 1)
            angle = t * 4.0 * pi + c * 2.0 * pi / float(classes)
            radius = 0.20 + 0.80 * t
            noise = 0.03 * math.sin(11.0 * t + c * 1.7)
            r = radius + noise
            x1 = r * math.cos(angle)
            x2 = r * math.sin(angle)
            x_data.append(x1)
            x_data.append(x2)
            y_data.append(float(c))
    return x_data, y_data, samples, classes

def argmax(probs, offset, classes):
    best_idx = 0
    best_val = probs[offset]
    for c in range(1, classes):
        val = probs[offset + c]
        if val > best_val:
            best_val = val
            best_idx = c
    return best_idx

def evaluate(model, x, y_data, samples, classes):
    out = model.forward(x)
    probs = out.to_vector()
    correct = 0
    for i in range(samples):
        pred = argmax(probs, i * classes, classes)
        truth = int(y_data[i])
        if pred == truth:
            correct += 1
    accuracy = (correct / float(samples)) * 100.0
    return out, accuracy

def main():
    print("==================================================")
    print("      LITETORCH PYTHON DEMO RUN BENCHMARK")
    print("==================================================")

    x_vec, y_vec, samples, classes = generate_spiral_data(600, 3)
    
    device = lt.auto_device()

    x_tensor = lt.Tensor.from_vector(x_vec, [samples, 2], device, False)
    y_tensor = lt.Tensor.from_vector(y_vec, [samples], device, False)

    l1 = lt.nn.Linear(2, 128, True)
    ln1 = lt.nn.LayerNorm([128])
    l2 = lt.nn.Linear(128, 128, True)
    ln2 = lt.nn.LayerNorm([128])
    l3 = lt.nn.Linear(128, 64, True)
    l4 = lt.nn.Linear(64, classes, True)

    class DeepSpiralNet(lt.nn.Module):
        def __init__(self, l1, ln1, l2, ln2, l3, l4):
            super().__init__()
            self.l1 = l1
            self.ln1 = ln1
            self.l2 = l2
            self.ln2 = ln2
            self.l3 = l3
            self.l4 = l4

        def forward(self, inp):
            h1 = self.l1.forward(inp)
            h1_norm = self.ln1.forward(h1)
            a1 = lt.Ops.gelu(h1_norm)
            h2 = self.l2.forward(a1)
            h2_norm = self.ln2.forward(h2)
            a2 = lt.Ops.gelu(h2_norm)
            h3 = self.l3.forward(a2)
            a3 = lt.Ops.gelu(h3)
            return self.l4.forward(a3)

        def parameters(self):
            return self.l1.parameters() + self.ln1.parameters() + self.l2.parameters() + self.ln2.parameters() + self.l3.parameters() + self.l4.parameters()

    net = DeepSpiralNet(l1, ln1, l2, ln2, l3, l4)
    optimizer = lt.optim.Adam(net.parameters(), lr=0.01)

    start_cpu = time.process_time()
    start_wall = time.perf_counter()

    out, acc = evaluate(net, x_tensor, y_vec, samples, classes)
    loss_init = lt.Ops.cross_entropy_loss(out, y_tensor)

    print(f"Dataset: {samples} samples | Classes: {classes}")
    print(f"Initial Loss: {loss_init.item():.6f} | Initial Accuracy: {acc:.2f}%")
    print(f"Initial RAM: {current_rss_mb():.2f} MB | Peak RAM: {peak_rss_mb():.2f} MB | VRAM: {get_vram_mb():.2f} MB")
    print("--------------------------------------------------")

    for epoch in range(1, 301):
        optimizer.zero_grad()
        out = net.forward(x_tensor)
        loss = lt.Ops.cross_entropy_loss(out, y_tensor)
        loss.backward()
        optimizer.step()

        if epoch % 50 == 0 or epoch == 1:
            out_eval, acc_eval = evaluate(net, x_tensor, y_vec, samples, classes)
            elapsed_cpu_ms = (time.process_time() - start_cpu) * 1000.0
            elapsed_wall_ms = (time.perf_counter() - start_wall) * 1000.0
            print(f"Epoch {epoch:3d} | Loss: {loss.item():.6f} | Acc: {acc_eval:.2f}% | RAM: {current_rss_mb():.2f} MB | Peak RAM: {peak_rss_mb():.2f} MB | VRAM: {get_vram_mb():.2f} MB | CPU Time: {elapsed_cpu_ms:.2f} ms | GPU Time: {elapsed_wall_ms:.2f} ms")

    final_cpu_ms = (time.process_time() - start_cpu) * 1000.0
    final_wall_ms = (time.perf_counter() - start_wall) * 1000.0

    print("--------------------------------------------------")
    print(f"Final Loss: {loss.item():.6f}")
    print(f"Final Accuracy: {acc_eval:.2f}%")
    print(f"Final RAM: {current_rss_mb():.2f} MB | Peak RAM: {peak_rss_mb():.2f} MB | VRAM: {get_vram_mb():.2f} MB")
    print(f"Total CPU Time: {final_cpu_ms:.2f} ms | Total GPU Time: {final_wall_ms:.2f} ms")
    print("==================================================")

if __name__ == "__main__":
    main()
