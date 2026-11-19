// LD_PRELOAD shim: counts HIP graph and allocation calls and the change of used card memory around each graph call.
// Prints one "HIPSHIM" line to stderr every HIPSHIM_EVERY seconds (default 10).
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

typedef int hipError_t;
typedef void *hipGraphExec_t, *hipGraph_t, *hipGraphNode_t, *hipStream_t;

static atomic_int hip_ready;
static hipError_t (*real_meminfo)(size_t *, size_t *);

// ggml loads its HIP backend with dlopen, so libamdhip64 is not in the global scope RTLD_NEXT searches.
static void *sym(const char *name) {
    static void *lib;
    void *p = dlsym(RTLD_NEXT, name);
    if (p) return p;
    if (!lib) {
        const char *so = getenv("HIPSHIM_LIB") ? getenv("HIPSHIM_LIB") : "libamdhip64.so.7";
        lib = dlopen(so, RTLD_LAZY | RTLD_NOLOAD);
        if (!lib) lib = dlopen(so, RTLD_LAZY);
    }
    p = lib ? dlsym(lib, name) : NULL;
    if (!p) { fprintf(stderr, "HIPSHIM cannot find %s\n", name); abort(); }
    return p;
}

static long long used_bytes(void) {
    size_t f = 0, t = 0;
    if (!real_meminfo) real_meminfo = sym("hipMemGetInfo");
    if (!real_meminfo || real_meminfo(&f, &t) != 0) return 0;
    return (long long)(t - f);
}

enum { C_INST, C_INSTF, C_UPD, C_UPDFAIL, C_EXECDESTROY, C_GRAPHDESTROY, C_BEGIN, C_END, C_LAUNCH, C_MALLOC, C_FREE, C_N };
static const char *names[C_N] = {"inst", "instf", "upd", "updfail", "execdes", "graphdes", "begin", "end", "launch", "malloc", "free"};
static atomic_llong cnt[C_N], dmem[C_N], tns[C_N];

static long long now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1000000000LL + ts.tv_nsec;
}
static atomic_llong live_execs, malloc_live;

#define REAL(name, ...) static hipError_t (*real)(__VA_ARGS__); if (!real) real = sym(name)
#define MEASURED(idx, call) long long u0 = used_bytes(), t0 = now_ns(); hipError_t r = call; \
    atomic_fetch_add(&tns[idx], now_ns() - t0); long long u1 = used_bytes(); \
    atomic_fetch_add(&cnt[idx], 1); atomic_fetch_add(&dmem[idx], u1 - u0); return r

hipError_t hipGraphInstantiate(hipGraphExec_t *e, hipGraph_t g, hipGraphNode_t *en, char *log, size_t n) {
    REAL("hipGraphInstantiate", hipGraphExec_t *, hipGraph_t, hipGraphNode_t *, char *, size_t);
    atomic_fetch_add(&live_execs, 1);
    MEASURED(C_INST, real(e, g, en, log, n));
}

hipError_t hipGraphInstantiateWithFlags(hipGraphExec_t *e, hipGraph_t g, unsigned long long fl) {
    REAL("hipGraphInstantiateWithFlags", hipGraphExec_t *, hipGraph_t, unsigned long long);
    atomic_fetch_add(&live_execs, 1);
    MEASURED(C_INSTF, real(e, g, fl));
}

hipError_t hipGraphExecUpdate(hipGraphExec_t e, hipGraph_t g, hipGraphNode_t *en, int *res) {
    REAL("hipGraphExecUpdate", hipGraphExec_t, hipGraph_t, hipGraphNode_t *, int *);
    long long u0 = used_bytes(), t0 = now_ns();
    hipError_t r = real(e, g, en, res);
    atomic_fetch_add(&tns[r == 0 ? C_UPD : C_UPDFAIL], now_ns() - t0);
    long long u1 = used_bytes();
    atomic_fetch_add(&cnt[r == 0 ? C_UPD : C_UPDFAIL], 1);
    atomic_fetch_add(&dmem[r == 0 ? C_UPD : C_UPDFAIL], u1 - u0);
    return r;
}

hipError_t hipGraphExecDestroy(hipGraphExec_t e) {
    REAL("hipGraphExecDestroy", hipGraphExec_t);
    atomic_fetch_sub(&live_execs, 1);
    MEASURED(C_EXECDESTROY, real(e));
}

hipError_t hipGraphDestroy(hipGraph_t g) {
    REAL("hipGraphDestroy", hipGraph_t);
    MEASURED(C_GRAPHDESTROY, real(g));
}

hipError_t hipStreamBeginCapture(hipStream_t s, int mode) {
    REAL("hipStreamBeginCapture", hipStream_t, int);
    atomic_fetch_add(&cnt[C_BEGIN], 1);
    return real(s, mode);
}

hipError_t hipStreamEndCapture(hipStream_t s, hipGraph_t *g) {
    REAL("hipStreamEndCapture", hipStream_t, hipGraph_t *);
    MEASURED(C_END, real(s, g));
}

hipError_t hipGraphLaunch(hipGraphExec_t e, hipStream_t s) {
    REAL("hipGraphLaunch", hipGraphExec_t, hipStream_t);
    long long t0 = now_ns();
    hipError_t r = real(e, s);
    atomic_fetch_add(&tns[C_LAUNCH], now_ns() - t0);
    atomic_fetch_add(&cnt[C_LAUNCH], 1);
    return r;
}

// pointer -> size, open addressing; frees of unknown pointers are only counted
#define TAB (1 << 16)
static void *tab_p[TAB];
static size_t tab_s[TAB];
static pthread_mutex_t tab_mu = PTHREAD_MUTEX_INITIALIZER;
static unsigned slot(void *p) { return (unsigned)(((uintptr_t)p >> 8) * 2654435761u) & (TAB - 1); }

hipError_t hipMalloc(void **p, size_t n) {
    REAL("hipMalloc", void **, size_t);
    hipError_t r = real(p, n);
    atomic_store(&hip_ready, 1);
    if (r == 0 && *p) {
        pthread_mutex_lock(&tab_mu);
        for (unsigned i = slot(*p), k = 0; k < TAB; i = (i + 1) & (TAB - 1), k++)
            if (!tab_p[i] || tab_p[i] == (void *)1) { tab_p[i] = *p; tab_s[i] = n; break; }
        pthread_mutex_unlock(&tab_mu);
        atomic_fetch_add(&cnt[C_MALLOC], 1);
        atomic_fetch_add(&malloc_live, (long long)n);
    }
    return r;
}

hipError_t hipFree(void *p) {
    REAL("hipFree", void *);
    if (p) {
        pthread_mutex_lock(&tab_mu);
        for (unsigned i = slot(p), k = 0; k < TAB && tab_p[i]; i = (i + 1) & (TAB - 1), k++)
            if (tab_p[i] == p) { atomic_fetch_sub(&malloc_live, (long long)tab_s[i]); tab_p[i] = (void *)1; break; }
        pthread_mutex_unlock(&tab_mu);
        atomic_fetch_add(&cnt[C_FREE], 1);
    }
    return real(p);
}

static void *reporter(void *arg) {
    (void)arg;
    const char *e = getenv("HIPSHIM_EVERY");
    int every = e ? atoi(e) : 10;
    for (;;) {
        sleep(every);
        if (!atomic_load(&hip_ready)) continue;
        char buf[1024];
        int o = snprintf(buf, sizeof buf, "HIPSHIM t=%ld used_mib=%.1f malloc_live_mib=%.1f live_execs=%lld",
                         (long)time(NULL), used_bytes() / 1048576.0, atomic_load(&malloc_live) / 1048576.0,
                         atomic_load(&live_execs));
        for (int i = 0; i < C_N; i++) {
            long long c = atomic_load(&cnt[i]), d = atomic_load(&dmem[i]);
            o += snprintf(buf + o, sizeof buf - o, " %s=%lld", names[i], c);
            if (d) o += snprintf(buf + o, sizeof buf - o, "/%+.2fMiB", d / 1048576.0);
            long long t = atomic_load(&tns[i]);
            if (t && c) o += snprintf(buf + o, sizeof buf - o, "/%.3fms", t / 1e6 / c);
        }
        fprintf(stderr, "%s\n", buf);
    }
    return NULL;
}

__attribute__((constructor)) static void start(void) {
    pthread_t t;
    pthread_create(&t, NULL, reporter, NULL);
    pthread_detach(t);
}
