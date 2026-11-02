#include "litetorch/quant_scheme.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <memory>
#include <string>
#include <vector>

using namespace litetorch;
using namespace litetorch::quantization;

static int passed = 0;
static int failed = 0;

static void check(const std::string& name, bool cond) {
    if (cond) {
        ++passed;
    } else {
        ++failed;
        printf("FAIL %s\n", name.c_str());
    }
}

static std::shared_ptr<Tensor> fvec(const std::vector<float>& v, const std::vector<int64_t>& shape) {
    return Tensor::from_vector(v, shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
}

static std::vector<float> tovec(std::shared_ptr<Tensor> t) {
    return t->to(Device(DeviceType::CPU, 0), DataType::FP32)->to_vector();
}

static bool near_vec(const std::vector<float>& a, const std::vector<float>& b, float tol) {
    if (a.size() != b.size()) return false;
    for (size_t i = 0; i < a.size(); ++i)
        if (std::fabs(a[i] - b[i]) > tol) return false;
    return true;
}

static bool throws_with(std::function<void()> fn, const std::string& needle) {
    try {
        fn();
    } catch (const std::exception& e) {
        return std::string(e.what()).find(needle) != std::string::npos;
    }
    return false;
}

static std::shared_ptr<nn::Linear> make_lin(int in_f, int out_f, const std::vector<float>& w, const std::vector<float>& b) {
    auto lin = std::make_shared<nn::Linear>(in_f, out_f, true);
    lin->weight = fvec(w, {(int64_t)out_f, (int64_t)in_f});
    lin->bias = fvec(b, {(int64_t)out_f});
    return lin;
}

static std::shared_ptr<nn::Sequential> make_seq() {
    auto seq = std::make_shared<nn::Sequential>();
    seq->add(make_lin(4, 3, {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f}, {0.1f, -0.1f, 0.2f}));
    seq->add(std::make_shared<nn::ReLU>());
    seq->add(make_lin(3, 2, {0.2f, -0.3f, 0.4f, -0.5f, 0.6f, -0.7f}, {0.05f, -0.05f}));
    return seq;
}

int main() {
    check("registry lists 4 built-ins", [&]() {
        auto names = list_schemes();
        bool ok = names.size() >= 4;
        for (const char* n : {"int8-per-channel", "dynamic-int8", "fake-quant", "nf4-block64"})
            if (!has_scheme(n)) ok = false;
        return ok;
    }());
    check("get unknown scheme throws named error", throws_with([]() { get_scheme("w54-no-such-scheme"); }, "not registered"));
    check("register and unregister custom scheme", [&]() {
        QuantScheme s;
        s.name = "w54-lifecycle";
        s.calibrate = [](std::shared_ptr<Tensor>, const Calibrator*, const std::string&, const std::map<std::string, std::string>&) { return QParams(); };
        s.quantize = [](std::shared_ptr<Tensor> t, const QParams&, const std::map<std::string, std::string>&) {
            QuantizedTensor r;
            r.data = t;
            r.logical_shape = t->shape;
            r.logical_dtype = t->dtype;
            return r;
        };
        s.dequantize = [](const QuantizedTensor& r) { return r.data; };
        register_scheme(s);
        bool a = has_scheme("w54-lifecycle");
        unregister_scheme("w54-lifecycle");
        bool b = !has_scheme("w54-lifecycle");
        return a && b;
    }());
    check("register scheme with empty name throws", throws_with([]() {
        QuantScheme s;
        register_scheme(s);
    }, "must not be empty"));
    check("custom affine scheme parity vs manual computation", [&]() {
        QuantScheme s;
        s.name = "w54-affine";
        s.calibrate = [](std::shared_ptr<Tensor> t, const Calibrator*, const std::string&, const std::map<std::string, std::string>&) {
            auto W = t->to(Device(DeviceType::CPU, 0), DataType::FP32);
            auto d = W->to_vector();
            float mn = d[0], mx = d[0];
            for (float v : d) { mn = std::min(mn, v); mx = std::max(mx, v); }
            QParams qp;
            qp.scale = (mx - mn) / 255.0f;
            if (qp.scale == 0.0f) qp.scale = 1.0f;
            qp.zero_point = std::round(-mn / qp.scale);
            if (qp.zero_point < 0.0f) qp.zero_point = 0.0f;
            if (qp.zero_point > 255.0f) qp.zero_point = 255.0f;
            qp.bits = 8;
            return qp;
        };
        s.quantize = [](std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
            auto W = t->to(Device(DeviceType::CPU, 0), DataType::FP32);
            auto d = W->to_vector();
            std::vector<float> q(d.size());
            for (size_t i = 0; i < d.size(); ++i) {
                float v = std::round(d[i] / qp.scale) + qp.zero_point;
                if (v < 0.0f) v = 0.0f;
                if (v > 255.0f) v = 255.0f;
                q[i] = v;
            }
            QuantizedTensor r;
            r.data = Tensor::from_vector(q, W->shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
            r.qparams = qp;
            r.logical_shape = t->shape;
            r.logical_dtype = t->dtype;
            return r;
        };
        s.dequantize = [](const QuantizedTensor& r) {
            auto d = r.data->to(Device(DeviceType::CPU, 0), DataType::FP32)->to_vector();
            std::vector<float> out(d.size());
            for (size_t i = 0; i < d.size(); ++i) out[i] = (d[i] - r.qparams.zero_point) * r.qparams.scale;
            return Tensor::from_vector(out, r.data->shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
        };
        register_scheme(s);
        auto x = fvec({-1.0f, -0.25f, 0.0f, 0.5f, 1.5f, 2.0f}, {2, 3});
        QuantScheme got = get_scheme("w54-affine");
        QParams qp = got.calibrate(x, nullptr, "", got.params);
        QuantizedTensor rec = got.quantize(x, qp, got.params);
        auto dq = got.dequantize(rec);
        auto got_v = tovec(dq);
        float mn = -1.0f, mx = 2.0f;
        float scale = (mx - mn) / 255.0f;
        float zp = std::round(-mn / scale);
        std::vector<float> expect;
        for (float v : {-1.0f, -0.25f, 0.0f, 0.5f, 1.5f, 2.0f}) {
            float q = std::round(v / scale) + zp;
            if (q < 0.0f) q = 0.0f;
            if (q > 255.0f) q = 255.0f;
            expect.push_back((q - zp) * scale);
        }
        bool ok = near_vec(got_v, expect, 1e-6f);
        unregister_scheme("w54-affine");
        return ok;
    }());
    check("default fake-quant equals dequantize(quantize(x))", [&]() {
        QuantScheme s;
        s.name = "w54-default-fq";
        s.calibrate = [](std::shared_ptr<Tensor> t, const Calibrator*, const std::string&, const std::map<std::string, std::string>&) {
            auto W = t->to(Device(DeviceType::CPU, 0), DataType::FP32);
            auto d = W->to_vector();
            float m = 0.0f;
            for (float v : d) m = std::max(m, std::fabs(v));
            QParams qp;
            qp.scale = (m == 0.0f) ? 1.0f : m / 127.0f;
            qp.bits = 8;
            return qp;
        };
        s.quantize = [](std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
            auto W = t->to(Device(DeviceType::CPU, 0), DataType::FP32);
            auto d = W->to_vector();
            std::vector<float> q(d.size());
            for (size_t i = 0; i < d.size(); ++i) {
                float v = std::round(d[i] / qp.scale);
                if (v > 127.0f) v = 127.0f;
                if (v < -127.0f) v = -127.0f;
                q[i] = v;
            }
            QuantizedTensor r;
            r.data = Tensor::from_vector(q, W->shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
            r.qparams = qp;
            r.logical_shape = t->shape;
            r.logical_dtype = t->dtype;
            return r;
        };
        s.dequantize = [](const QuantizedTensor& r) {
            auto d = r.data->to(Device(DeviceType::CPU, 0), DataType::FP32)->to_vector();
            std::vector<float> out(d.size());
            for (size_t i = 0; i < d.size(); ++i) out[i] = d[i] * r.qparams.scale;
            return Tensor::from_vector(out, r.data->shape, Device(DeviceType::CPU, 0), false, DataType::FP32);
        };
        register_scheme(s);
        QuantScheme got = get_scheme("w54-default-fq");
        auto x = fvec({0.3f, -0.9f, 1.2f, 0.05f}, {2, 2});
        QParams qp = got.calibrate(x, nullptr, "", got.params);
        auto fq = scheme_fake_quant(got, x, qp, got.params);
        auto direct = got.dequantize(got.quantize(x, qp, got.params));
        bool ok = near_vec(tovec(fq), tovec(direct), 0.0f);
        unregister_scheme("w54-default-fq");
        return ok;
    }());
    check("int8-per-channel scales and codes match manual computation", [&]() {
        QuantScheme s = get_scheme("int8-per-channel");
        auto W = fvec({0.5f, -1.0f, 0.25f, -0.2f, 0.4f, 0.1f}, {2, 3});
        QParams qp = s.calibrate(W, nullptr, "", s.params);
        auto scales = tovec(qp.scales);
        bool ok = scales.size() == 2 && std::fabs(scales[0] - 1.0f / 127.0f) < 1e-7f && std::fabs(scales[1] - 0.4f / 127.0f) < 1e-7f;
        QuantizedTensor rec = s.quantize(W, qp, s.params);
        auto qdata = (const int8_t*)rec.data->storage->get_cpu_ptr();
        float orig6[6] = {0.5f, -1.0f, 0.25f, -0.2f, 0.4f, 0.1f};
        ok = ok && rec.data->dtype == DataType::INT8;
        for (int i = 0; i < 6; ++i) {
            float sc = (i < 3) ? scales[0] : scales[1];
            float v = std::round(orig6[i] / sc);
            if (v > 127.0f) v = 127.0f;
            if (v < -127.0f) v = -127.0f;
            if (qdata[i] != (int8_t)v) ok = false;
        }
        auto dq = s.dequantize(rec);
        auto dqv = tovec(dq);
        for (size_t i = 0; i < 6; ++i) {
            float sc = (i < 3) ? scales[0] : scales[1];
            if (std::fabs(dqv[i] - orig6[i]) > sc / 2.0f + 1e-6f) ok = false;
        }
        return ok;
    }());
    check("nf4-block64 scales are block maxabs and grid points exact", [&]() {
        QuantScheme s = get_scheme("nf4-block64");
        std::vector<float> vals(130, 0.0f);
        vals[0] = 0.8f;
        vals[63] = -0.4f;
        vals[64] = 0.2f;
        vals[127] = -0.1f;
        vals[128] = 1.0f;
        vals[129] = -1.0f;
        auto W = fvec(vals, {130});
        QParams qp = s.calibrate(W, nullptr, "", s.params);
        auto scales = tovec(qp.scales);
        bool ok = scales.size() == 3 && std::fabs(scales[0] - 0.8f) < 1e-7f && std::fabs(scales[1] - 0.2f) < 1e-7f && std::fabs(scales[2] - 1.0f) < 1e-7f;
        QuantizedTensor rec = s.quantize(W, qp, s.params);
        ok = ok && rec.data->dtype == DataType::NF4;
        auto dq = s.dequantize(rec);
        auto dqv = tovec(dq);
        if (std::fabs(dqv[0] - 0.8f) > 1e-6f) ok = false;
        if (std::fabs(dqv[128] - 1.0f) > 1e-6f) ok = false;
        if (std::fabs(dqv[129] + 1.0f) > 1e-6f) ok = false;
        for (size_t i = 0; i < 130; ++i) {
            float sc = scales[i / 64];
            if (std::fabs(dqv[i] - vals[i]) > 0.16f * sc + 1e-6f) ok = false;
        }
        return ok;
    }());
    check("walker replaces Linears and keeps ReLU", [&]() {
        auto seq = make_seq();
        auto out_model = quantize_model(seq, "int8-per-channel");
        auto kids = out_model->named_children();
        bool ok = kids.size() == 3;
        if (ok) {
            ok = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second) != nullptr;
            ok = ok && std::dynamic_pointer_cast<nn::ReLU>(kids[1].second) != nullptr;
            ok = ok && std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[2].second) != nullptr;
        }
        return ok;
    }());
    check("walker forward close to float forward", [&]() {
        auto seq = make_seq();
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto ref = tovec(seq->forward(x));
        auto qseq = quantize_model(seq, "int8-per-channel");
        auto got = tovec(qseq->forward(x));
        return near_vec(got, ref, 0.05f);
    }());
    check("quantized linear parameters returns only bias", [&]() {
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "int8-per-channel");
        auto kids = qseq->named_children();
        auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second);
        if (!qlin) return false;
        auto params = qlin->parameters();
        return params.size() == 1 && params[0] == qlin->bias;
    }());
    check("state_dict carries record tensors", [&]() {
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "int8-per-channel");
        auto sd = qseq->state_dict();
        bool has_data = false, has_scales = false, has_bias = false;
        for (auto& kv : sd) {
            if (kv.first == "0.record_data") has_data = true;
            if (kv.first == "0.record_scales") has_scales = true;
            if (kv.first == "0.bias") has_bias = true;
        }
        return has_data && has_scales && has_bias;
    }());
    check("dynamic-int8 walker sets dynamic activation", [&]() {
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "dynamic-int8");
        auto kids = qseq->named_children();
        auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second);
        if (!qlin) return false;
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto out = qseq->forward(x);
        return qlin->dynamic_activation && out != nullptr;
    }());
    check("walker fake-quantizes LSTM cell weights in place", [&]() {
        auto lstm = std::make_shared<nn::LSTM>(4, 3, 1, true);
        auto before = tovec(lstm->cells[0]->weight_ih);
        bool any_off_grid = false;
        for (float v : before) if (std::fabs(v) > 1e-6f) any_off_grid = true;
        if (!any_off_grid) return false;
        quantize_model(lstm, "fake-quant");
        auto after = tovec(lstm->cells[0]->weight_ih);
        if (after.size() != before.size()) return false;
        float mn = before[0], mx = before[0];
        for (float v : before) { mn = std::min(mn, v); mx = std::max(mx, v); }
        float bound = 127.0f;
        float scale = (mx - mn) / (2.0f * bound);
        if (scale == 0.0f) scale = 1.0f;
        float zp = std::round(-bound - mn / scale);
        if (zp < -bound) zp = -bound;
        if (zp > bound) zp = bound;
        for (size_t i = 0; i < before.size(); ++i) {
            float q = std::round(before[i] / scale + zp);
            if (q < -bound) q = -bound;
            if (q > bound) q = bound;
            float expect = (q - zp) * scale;
            if (std::fabs(after[i] - expect) > 1e-5f) return false;
        }
        return true;
    }());
    check("PTQ tier refuses uncalibrated prepared model", [&]() {
        auto seq = make_seq();
        auto prepared = prepare(seq, PTQConfig());
        return throws_with([&]() { quantize_ptq(prepared, "fake-quant"); }, "calibrate");
    }());
    check("PTQ tier calibrates then quantizes", [&]() {
        auto seq = make_seq();
        auto prepared = prepare(seq, PTQConfig());
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        prepared->calibrate([x]() { return x; }, 2);
        bool has_cal = !prepared->calibrator.min_vals.empty();
        auto out_model = quantize_ptq(prepared, "fake-quant");
        auto kids = out_model->named_children();
        bool ok = has_cal && kids.size() == 3;
        ok = ok && std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second) != nullptr;
        auto out = out_model->forward(x);
        return ok && out != nullptr;
    }());
    check("QAT wrapper eval forward equals manual matmul plus bias", [&]() {
        auto lin = make_lin(4, 3, {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f}, {0.1f, -0.1f, 0.2f});
        auto qat = std::make_shared<SchemeQATLinear>(lin, "int8-per-channel");
        if (!qat->weight_qparams.scales) return false;
        qat->eval();
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto got = tovec(qat->forward(x));
        auto W = lin->weight;
        auto manual = Ops::add(Ops::matmul(x, W->transpose(0, 1)), lin->bias);
        return near_vec(got, tovec(manual), 1e-6f);
    }());
    check("QAT wrapper training forward runs", [&]() {
        auto lin = make_lin(4, 3, {0.1f, -0.2f, 0.3f, -0.4f, 0.5f, -0.6f, 0.7f, -0.8f, 0.9f, -1.0f, 1.1f, -1.2f}, {0.1f, -0.1f, 0.2f});
        auto qat = std::make_shared<SchemeQATLinear>(lin, "fake-quant");
        qat->train();
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto out = qat->forward(x);
        return out != nullptr && qat->act_calibrated;
    }());
    check("save and load round-trip preserves forward exactly", [&]() {
        (void)std::system("rm -rf core_w54_tmp");
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "int8-per-channel");
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto ref = tovec(qseq->forward(x));
        save_quantized(qseq, "core_w54_tmp");
        auto fresh = make_seq();
        auto loaded = load_quantized(fresh, "core_w54_tmp");
        auto got = tovec(loaded->forward(x));
        auto kids = loaded->named_children();
        bool ok = near_vec(got, ref, 0.0f);
        ok = ok && kids.size() == 3;
        if (ok) {
            auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second);
            ok = qlin != nullptr && qlin->record.scheme_name == "int8-per-channel";
        }
        (void)std::system("rm -rf core_w54_tmp");
        return ok;
    }());
    check("nf4 save and load round-trip preserves forward exactly", [&]() {
        (void)std::system("rm -rf core_w54_tmp3");
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "nf4-block64");
        auto x = fvec({0.5f, -0.5f, 1.0f, -1.0f}, {1, 4});
        auto ref = tovec(qseq->forward(x));
        save_quantized(qseq, "core_w54_tmp3");
        auto fresh = std::make_shared<nn::Sequential>();
        fresh->add(make_lin(4, 3, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f, 0.0f}));
        fresh->add(std::make_shared<nn::ReLU>());
        fresh->add(make_lin(3, 2, {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, {0.0f, 0.0f}));
        auto loaded = load_quantized(fresh, "core_w54_tmp3");
        auto got = tovec(loaded->forward(x));
        auto kids = loaded->named_children();
        bool ok = near_vec(got, ref, 0.0f);
        ok = ok && kids.size() == 3;
        if (ok) {
            auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second);
            ok = qlin != nullptr && qlin->record.scheme_name == "nf4-block64" && qlin->record.data->dtype == DataType::NF4;
        }
        (void)std::system("rm -rf core_w54_tmp3");
        return ok;
    }());
    check("load requires re-registration of custom scheme", [&]() {
        QuantScheme s;
        s.name = "w54-needs-rereg";
        s.calibrate = [](std::shared_ptr<Tensor> t, const Calibrator*, const std::string&, const std::map<std::string, std::string>&) {
            QParams qp;
            qp.scale = 0.01f;
            qp.bits = 8;
            (void)t;
            return qp;
        };
        s.quantize = [](std::shared_ptr<Tensor> t, const QParams& qp, const std::map<std::string, std::string>&) {
            QuantizedTensor r;
            r.data = Ops::fake_quantize(t, qp.scale, 0.0f, 8);
            r.qparams = qp;
            r.logical_shape = t->shape;
            r.logical_dtype = t->dtype;
            return r;
        };
        s.dequantize = [](const QuantizedTensor& r) { return r.data; };
        register_scheme(s);
        (void)std::system("rm -rf core_w54_tmp2");
        auto seq = make_seq();
        auto qseq = quantize_model(seq, "w54-needs-rereg");
        save_quantized(qseq, "core_w54_tmp2");
        unregister_scheme("w54-needs-rereg");
        auto fresh = make_seq();
        bool threw = throws_with([&]() { load_quantized(fresh, "core_w54_tmp2"); }, "not registered");
        register_scheme(s);
        auto loaded = load_quantized(fresh, "core_w54_tmp2");
        auto kids = loaded->named_children();
        bool ok = threw && kids.size() == 3;
        if (ok) {
            auto qlin = std::dynamic_pointer_cast<SchemeQuantizedLinear>(kids[0].second);
            ok = qlin != nullptr && qlin->record.scheme_name == "w54-needs-rereg";
        }
        unregister_scheme("w54-needs-rereg");
        (void)std::system("rm -rf core_w54_tmp2");
        return ok;
    }());
    check("LSTM cell save and load round-trip preserves forward exactly and biases", [&]() {
        (void)std::system("rm -rf core_w59_tmp_lstm");
        auto lstm = std::make_shared<nn::LSTM>(8, 16, 1, true);
        auto qlstm = quantize_model(lstm, "fake-quant");
        std::vector<float> xv;
        for (int i = 0; i < 5 * 2 * 8; ++i) xv.push_back(std::sin(i * 0.37f));
        auto x = fvec(xv, {5, 2, 8});
        auto ref = tovec(qlstm->forward(x));
        save_quantized(qlstm, "core_w59_tmp_lstm");
        auto fresh = std::make_shared<nn::LSTM>(8, 16, 1, true);
        auto loaded = load_quantized(fresh, "core_w59_tmp_lstm");
        auto got = tovec(loaded->forward(x));
        bool ok = near_vec(got, ref, 0.0f);
        auto qcell = std::dynamic_pointer_cast<nn::LSTM>(qlstm)->cells[0];
        auto lcell = std::dynamic_pointer_cast<nn::LSTM>(loaded)->cells[0];
        ok = ok && near_vec(tovec(lcell->bias_ih), tovec(qcell->bias_ih), 0.0f);
        ok = ok && near_vec(tovec(lcell->bias_hh), tovec(qcell->bias_hh), 0.0f);
        ok = ok && near_vec(tovec(lcell->weight_ih), tovec(qcell->weight_ih), 0.0f);
        ok = ok && near_vec(tovec(lcell->weight_hh), tovec(qcell->weight_hh), 0.0f);
        for (auto& kv : lcell->named_parameters()) {
            if (kv.first == "weight_ih") ok = ok && kv.second == lcell->weight_ih;
            if (kv.first == "weight_hh") ok = ok && kv.second == lcell->weight_hh;
            if (kv.first == "bias_ih") ok = ok && kv.second == lcell->bias_ih;
            if (kv.first == "bias_hh") ok = ok && kv.second == lcell->bias_hh;
        }
        (void)std::system("rm -rf core_w59_tmp_lstm");
        return ok;
    }());
    check("GRU cell save and load round-trip preserves forward exactly and biases", [&]() {
        (void)std::system("rm -rf core_w59_tmp_gru");
        auto gru = std::make_shared<nn::GRU>(8, 16, 1, true);
        auto qgru = quantize_model(gru, "fake-quant");
        std::vector<float> xv;
        for (int i = 0; i < 5 * 2 * 8; ++i) xv.push_back(std::sin(i * 0.37f));
        auto x = fvec(xv, {5, 2, 8});
        auto ref = tovec(qgru->forward(x));
        save_quantized(qgru, "core_w59_tmp_gru");
        auto fresh = std::make_shared<nn::GRU>(8, 16, 1, true);
        auto loaded = load_quantized(fresh, "core_w59_tmp_gru");
        auto got = tovec(loaded->forward(x));
        bool ok = near_vec(got, ref, 0.0f);
        auto qcell = std::dynamic_pointer_cast<nn::GRU>(qgru)->cells[0];
        auto lcell = std::dynamic_pointer_cast<nn::GRU>(loaded)->cells[0];
        ok = ok && near_vec(tovec(lcell->bias_ih), tovec(qcell->bias_ih), 0.0f);
        ok = ok && near_vec(tovec(lcell->bias_hh), tovec(qcell->bias_hh), 0.0f);
        ok = ok && near_vec(tovec(lcell->weight_ih), tovec(qcell->weight_ih), 0.0f);
        ok = ok && near_vec(tovec(lcell->weight_hh), tovec(qcell->weight_hh), 0.0f);
        for (auto& kv : lcell->named_parameters()) {
            if (kv.first == "weight_ih") ok = ok && kv.second == lcell->weight_ih;
            if (kv.first == "weight_hh") ok = ok && kv.second == lcell->weight_hh;
            if (kv.first == "bias_ih") ok = ok && kv.second == lcell->bias_ih;
            if (kv.first == "bias_hh") ok = ok && kv.second == lcell->bias_hh;
        }
        (void)std::system("rm -rf core_w59_tmp_gru");
        return ok;
    }());
    check("bias-free LSTM cell save and load round-trip preserves forward exactly", [&]() {
        (void)std::system("rm -rf core_w59_tmp_nobias");
        auto lstm = std::make_shared<nn::LSTM>(8, 16, 1, false);
        auto qlstm = quantize_model(lstm, "fake-quant");
        std::vector<float> xv;
        for (int i = 0; i < 5 * 2 * 8; ++i) xv.push_back(std::sin(i * 0.37f));
        auto x = fvec(xv, {5, 2, 8});
        auto ref = tovec(qlstm->forward(x));
        save_quantized(qlstm, "core_w59_tmp_nobias");
        auto fresh = std::make_shared<nn::LSTM>(8, 16, 1, false);
        auto loaded = load_quantized(fresh, "core_w59_tmp_nobias");
        auto got = tovec(loaded->forward(x));
        auto lcell = std::dynamic_pointer_cast<nn::LSTM>(loaded)->cells[0];
        bool ok = near_vec(got, ref, 0.0f) && lcell->bias_ih == nullptr && lcell->bias_hh == nullptr;
        (void)std::system("rm -rf core_w59_tmp_nobias");
        return ok;
    }());
    check("bias-free GRU cell save and load round-trip preserves forward exactly", [&]() {
        (void)std::system("rm -rf core_w59_tmp_nobias_gru");
        auto gru = std::make_shared<nn::GRU>(8, 16, 1, false);
        auto qgru = quantize_model(gru, "fake-quant");
        std::vector<float> xv;
        for (int i = 0; i < 5 * 2 * 8; ++i) xv.push_back(std::sin(i * 0.37f));
        auto x = fvec(xv, {5, 2, 8});
        auto ref = tovec(qgru->forward(x));
        save_quantized(qgru, "core_w59_tmp_nobias_gru");
        auto fresh = std::make_shared<nn::GRU>(8, 16, 1, false);
        auto loaded = load_quantized(fresh, "core_w59_tmp_nobias_gru");
        auto got = tovec(loaded->forward(x));
        auto lcell = std::dynamic_pointer_cast<nn::GRU>(loaded)->cells[0];
        bool ok = near_vec(got, ref, 0.0f) && lcell->bias_ih == nullptr && lcell->bias_hh == nullptr;
        (void)std::system("rm -rf core_w59_tmp_nobias_gru");
        return ok;
    }());
    printf("core_w54: %d passed, %d failed\n", passed, failed);
    return failed == 0 ? 0 : 1;
}
