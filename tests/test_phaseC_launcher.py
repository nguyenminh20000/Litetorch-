import importlib.util
import json
import os
import socket
import subprocess
import sys
import tempfile
import threading

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LT_RUN = os.path.join(REPO, "tools", "lt-run")

results = []


def check(name, cond):
    results.append((name, bool(cond)))
    print("[%s] %s" % ("PASS" if cond else "FAIL", name), flush=True)


def free_port():
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def write_worker(tmpd, name="worker.py", body=None):
    path = os.path.join(tmpd, name)
    if body is None:
        body = (
            "import json, os\n"
            "d = {k: os.environ.get(k, '') for k in (\n"
            "    'RANK', 'LOCAL_RANK', 'WORLD_SIZE', 'LOCAL_WORLD_SIZE',\n"
            "    'MASTER_ADDR', 'MASTER_PORT', 'NODE_RANK', 'NNODES',\n"
            "    'GROUP_RANK', 'ROLE_RANK')}\n"
            "open(os.environ['OUT_FILE'] + '.' + d['RANK'], 'w').write(json.dumps(d))\n"
        )
    with open(path, "w") as f:
        f.write(body)
    return path


def run_lt_run(args, env_extra=None, timeout=90):
    env = dict(os.environ)
    env.update(env_extra or {})
    return subprocess.run(
        [sys.executable, LT_RUN] + args, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, timeout=timeout)


def test_standalone_env():
    with tempfile.TemporaryDirectory() as tmpd:
        worker = write_worker(tmpd)
        rc = run_lt_run([
            "--standalone", "--nproc-per-node=2", "--no-prefix",
            worker,
        ], env_extra={"OUT_FILE": os.path.join(tmpd, "env")})
        check("standalone exit 0", rc.returncode == 0)
        envs = []
        for r in ("0", "1"):
            p = os.path.join(tmpd, "env." + r)
            if os.path.exists(p):
                with open(p) as f:
                    envs.append(json.load(f))
        check("standalone 2 workers ran", len(envs) == 2)
        if len(envs) == 2:
            ranks = sorted(e["RANK"] for e in envs)
            check("standalone ranks 0,1", ranks == ["0", "1"])
            check("standalone world_size 2", all(e["WORLD_SIZE"] == "2" for e in envs))
            check("standalone local_ranks 0,1",
                  sorted(e["LOCAL_RANK"] for e in envs) == ["0", "1"])
            check("standalone node_rank 0", all(e["NODE_RANK"] == "0" for e in envs))
            check("standalone nnodes 1", all(e["NNODES"] == "1" for e in envs))
            check("standalone master_port set",
                  all(e["MASTER_PORT"] not in ("", "0") for e in envs))


def test_multinode_rendezvous():
    port = free_port()
    with tempfile.TemporaryDirectory() as tmpd:
        worker = write_worker(tmpd)
        out = os.path.join(tmpd, "env")
        node0 = subprocess.Popen(
            [sys.executable, LT_RUN, "--nnodes=2", "--nproc-per-node=2",
             "--node-rank=0", "--master-addr=127.0.0.1",
             "--master-port=%d" % port, "--no-prefix", worker],
            env={**os.environ, "OUT_FILE": out},
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            rc1 = run_lt_run([
                "--nnodes=2", "--nproc-per-node=2",
                "--node-rank=1", "--master-addr=127.0.0.1",
                "--master-port=%d" % port, "--no-prefix", worker,
            ], env_extra={"OUT_FILE": out}, timeout=90)
            out0, err0 = node0.communicate(timeout=60)
        finally:
            if node0.poll() is None:
                node0.kill()
        check("multinode node0 exit 0", node0.returncode == 0)
        check("multinode node1 exit 0", rc1.returncode == 0)
        envs = []
        for r in ("0", "1", "2", "3"):
            p = out + "." + r
            if os.path.exists(p):
                with open(p) as f:
                    envs.append(json.load(f))
        check("multinode 4 workers ran", len(envs) == 4)
        if len(envs) == 4:
            by_rank = {e["RANK"]: e for e in envs}
            check("multinode world_size 4",
                  all(e["WORLD_SIZE"] == "4" for e in envs))
            check("multinode node0 ranks 0,1 local 0,1",
                  by_rank["0"]["NODE_RANK"] == "0" and by_rank["1"]["NODE_RANK"] == "0" and
                  by_rank["0"]["LOCAL_RANK"] == "0" and by_rank["1"]["LOCAL_RANK"] == "1")
            check("multinode node1 ranks 2,3 local 0,1",
                  by_rank["2"]["NODE_RANK"] == "1" and by_rank["3"]["NODE_RANK"] == "1" and
                  by_rank["2"]["LOCAL_RANK"] == "0" and by_rank["3"]["LOCAL_RANK"] == "1")
            check("multinode nnodes 2", all(e["NNODES"] == "2" for e in envs))


def test_failure_propagation():
    with tempfile.TemporaryDirectory() as tmpd:
        worker = write_worker(tmpd, name="fail.py", body=(
            "import os, sys, time\n"
            "time.sleep(0.2 if os.environ['RANK'] == '0' else 5)\n"
            "sys.exit(3)\n"
        ))
        rc = run_lt_run([
            "--standalone", "--nproc-per-node=2", "--no-prefix", worker,
        ], timeout=60)
        check("failure propagates exit code 3", rc.returncode == 3)


def load_lt_run_module():
    from importlib.machinery import SourceFileLoader
    loader = SourceFileLoader("ltrun", LT_RUN)
    spec = importlib.util.spec_from_loader("ltrun", loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    return mod


CPP_DRIVER_CLIENT = r"""
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    if (argc != 6) return 2;
    try {
        auto info = litetorch::distributed::launcher::rendezvous_join(
            argv[1], std::stoi(argv[2]), std::stoi(argv[3]),
            std::stoi(argv[4]), std::stoi(argv[5]), 30);
        std::printf("{\"rank_offset\":%d,\"world_size\":%d,\"node_rank\":%d,\"nnodes\":%d}\n",
            info.rank_offset, info.world_size, info.node_rank, info.nnodes);
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
"""

CPP_DRIVER_SERVER = r"""
#include <cstdio>
#include <string>
int main(int argc, char** argv) {
    if (argc != 3) return 2;
    try {
        litetorch::distributed::launcher::RendezvousServer srv(
            "0.0.0.0", std::stoi(argv[1]), std::stoi(argv[2]), 60);
        std::printf("PORT %d\n", srv.port());
        std::fflush(stdout);
        auto res = srv.run();
        std::printf("WORLD %d NODES %d\n", res.world_size, (int)res.rank_offsets.size());
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
}
"""

def build_cpp_drivers(tmpd):
    src = os.path.join(REPO, "src", "distributed", "launcher.cpp")
    bins = {}
    for name, driver in (("client", CPP_DRIVER_CLIENT), ("server", CPP_DRIVER_SERVER)):
        dsrc = os.path.join(tmpd, "drv_%s.cpp" % name)
        with open(dsrc, "w") as f:
            f.write('#include "%s"\n' % src + driver)
        dbin = os.path.join(tmpd, "drv_%s" % name)
        r = subprocess.run(
            ["g++", "-std=c++17", "-O1", dsrc, "-o", dbin, "-pthread"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=180)
        if r.returncode != 0:
            print(r.stderr[-2000:])
            return None, None
        bins[name] = dbin
    return bins["client"], bins["server"]


def test_cpp_rendezvous():
    with tempfile.TemporaryDirectory() as tmpd:
        client, server = build_cpp_drivers(tmpd)
        check("cpp drivers built", client is not None)
        if client is None:
            return
        port = free_port()
        srv = subprocess.Popen(
            [server, str(port), "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = srv.stdout.readline()
            check("cpp server bound", line.startswith("PORT"))
            p0 = subprocess.Popen(
                [client, "127.0.0.1", str(port), "0", "2", "2"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            p1 = subprocess.Popen(
                [client, "127.0.0.1", str(port), "1", "3", "2"],
                stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            try:
                o0, e0 = p0.communicate(timeout=60)
                o1, e1 = p1.communicate(timeout=60)
            finally:
                for p in (p0, p1):
                    if p.poll() is None:
                        p.kill()
            check("cpp client0 ok", p0.returncode == 0)
            check("cpp client1 ok", p1.returncode == 0)
            if p0.returncode == 0 and p1.returncode == 0:
                j0 = json.loads(o0)
                j1 = json.loads(o1)
                check("cpp offsets 0 and 2",
                      j0["rank_offset"] == 0 and j1["rank_offset"] == 2)
                check("cpp world_size 5",
                      j0["world_size"] == 5 and j1["world_size"] == 5)
            srv.wait(timeout=60)
            check("cpp server exit 0", srv.returncode == 0)
        finally:
            if srv.poll() is None:
                srv.kill()


def test_proto_py_server_cpp_client():
    mod = load_lt_run_module()
    with tempfile.TemporaryDirectory() as tmpd:
        client, _ = build_cpp_drivers(tmpd)
        check("cpp client built", client is not None)
        if client is None:
            return
        srv = mod.RendezvousServer(nnodes=2, timeout=30)
        port = srv.port
        t = threading.Thread(target=srv.run, daemon=True)
        t.start()
        p0 = subprocess.Popen(
            [client, "127.0.0.1", str(port), "0", "2", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        p1 = subprocess.Popen(
            [client, "127.0.0.1", str(port), "1", "1", "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            o0, _ = p0.communicate(timeout=60)
            o1, _ = p1.communicate(timeout=60)
        finally:
            for p in (p0, p1):
                if p.poll() is None:
                    p.kill()
        t.join(timeout=30)
        check("py-server/cpp-client c0 ok", p0.returncode == 0)
        check("py-server/cpp-client c1 ok", p1.returncode == 0)
        if p0.returncode == 0 and p1.returncode == 0:
            j0 = json.loads(o0)
            j1 = json.loads(o1)
            check("py-server/cpp-client offsets",
                  j0["rank_offset"] == 0 and j1["rank_offset"] == 2 and
                  j0["world_size"] == 3)


def test_proto_cpp_server_py_client():
    mod = load_lt_run_module()
    with tempfile.TemporaryDirectory() as tmpd:
        _, server = build_cpp_drivers(tmpd)
        check("cpp server built", server is not None)
        if server is None:
            return
        port = free_port()
        srv = subprocess.Popen(
            [server, str(port), "2"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        try:
            line = srv.stdout.readline()
            if not line.startswith("PORT"):
                check("cpp server bound", False)
                return
            results = {}
            errs = []

            def join_node(nr, nproc):
                try:
                    results[nr] = mod.rendezvous_client(
                        "127.0.0.1", port, nr, nproc, 2, 30)
                except Exception as e:  # noqa: BLE001
                    errs.append(str(e))

            ts = [threading.Thread(target=join_node, args=(0, 2)),
                  threading.Thread(target=join_node, args=(1, 1))]
            for t in ts:
                t.start()
            for t in ts:
                t.join(timeout=60)
            check("cpp-server/py-client no errors", not errs)
            if not errs and len(results) == 2:
                check("cpp-server/py-client offsets",
                      results[0]["rank_offset"] == 0 and
                      results[1]["rank_offset"] == 2 and
                      results[0]["world_size"] == 3)
            srv.wait(timeout=60)
            check("cpp server exit 0", srv.returncode == 0)
        finally:
            if srv.poll() is None:
                srv.kill()


if __name__ == "__main__":
    test_standalone_env()
    test_multinode_rendezvous()
    test_failure_propagation()
    test_cpp_rendezvous()
    test_proto_py_server_cpp_client()
    test_proto_cpp_server_py_client()
    n_fail = sum(1 for _, ok in results if not ok)
    print("----")
    print("%d/%d passed" % (len(results) - n_fail, len(results)))
    sys.exit(1 if n_fail else 0)
