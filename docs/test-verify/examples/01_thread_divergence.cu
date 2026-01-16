/**
 * Example 1: Thread Divergence in eBPF Hook - XDP-style Packet Parsing
 *
 * This example simulates an XDP-like eBPF program that:
 *   1. Parses Ethernet header (L2)
 *   2. Checks if IPv4 (L3)
 *   3. Checks if TCP (L4)
 *   4. Checks if HTTP (port 80)
 *   5. Parses HTTP path and updates path counters
 *
 * The eBPF hook would PASS traditional eBPF verification:
 *   ✓ Memory safe (bounds checked before access)
 *   ✓ Bounded execution (finite parsing depth)
 *   ✓ Valid helper usage
 *
 * But causes GPU-specific issues:
 *   ✗ Massive warp divergence from packet type branching
 *   ✗ Different threads parse different protocol layers
 *   ✗ HTTP path matching causes further divergence
 */

#include <stdio.h>
#include <cuda_runtime.h>
#include <string.h>

//=============================================================================
// Network Protocol Definitions (like in real XDP/eBPF)
//=============================================================================

#define ETH_P_IP    0x0800
#define IPPROTO_TCP 6
#define IPPROTO_UDP 17
#define HTTP_PORT   80

// Ethernet header
struct ethhdr {
    unsigned char  h_dest[6];
    unsigned char  h_source[6];
    unsigned short h_proto;
};

// IPv4 header (simplified)
struct iphdr {
    unsigned char  ihl_version;  // version:4, ihl:4
    unsigned char  tos;
    unsigned short tot_len;
    unsigned short id;
    unsigned short frag_off;
    unsigned char  ttl;
    unsigned char  protocol;
    unsigned short check;
    unsigned int   saddr;
    unsigned int   daddr;
};

// TCP header (simplified)
struct tcphdr {
    unsigned short source;
    unsigned short dest;
    unsigned int   seq;
    unsigned int   ack_seq;
    unsigned short flags;  // data offset, flags
    unsigned short window;
    unsigned short check;
    unsigned short urg_ptr;
};

//=============================================================================
// Simulated eBPF Infrastructure
//=============================================================================

// BPF map for path counters (like BPF_MAP_TYPE_HASH)
#define MAX_PATHS 8
#define PATH_API_USERS    0  // /api/users
#define PATH_API_ORDERS   1  // /api/orders
#define PATH_API_PRODUCTS 2  // /api/products
#define PATH_STATIC       3  // /static/*
#define PATH_INDEX        4  // /index.html
#define PATH_HEALTH       5  // /health
#define PATH_METRICS      6  // /metrics
#define PATH_OTHER        7  // other paths

__device__ unsigned long long path_counters[MAX_PATHS];
__device__ unsigned long long protocol_counters[4];  // [0]=non-ip, [1]=non-tcp, [2]=non-http, [3]=http

// eBPF Helper: Get thread index (packet index)
__device__ void bpf_get_thread_idx(unsigned long long *x, unsigned long long *y, unsigned long long *z) {
    *x = threadIdx.x + blockIdx.x * blockDim.x;
    *y = threadIdx.y;
    *z = threadIdx.z;
}

// eBPF Helper: Atomic increment (for counters)
__device__ void bpf_map_atomic_inc(unsigned long long *counter) {
    atomicAdd(counter, 1ULL);
}

// eBPF Helper: Bounds check (like in XDP)
__device__ int bpf_check_bounds(void *data, void *data_end, void *ptr, int size) {
    return ((unsigned char*)ptr + size <= (unsigned char*)data_end);
}

//=============================================================================
// Simulated Packet Data
//=============================================================================

#define PACKET_SIZE 256
#define HTTP_PAYLOAD_OFFSET (sizeof(struct ethhdr) + sizeof(struct iphdr) + sizeof(struct tcphdr))

// Generate simulated packet with given characteristics
__device__ void generate_packet(unsigned char *pkt, int pkt_idx, int *pkt_type) {
    // Vary packet types based on index to simulate real traffic
    int type = pkt_idx % 10;

    struct ethhdr *eth = (struct ethhdr *)pkt;
    struct iphdr *ip = (struct iphdr *)(pkt + sizeof(struct ethhdr));
    struct tcphdr *tcp = (struct tcphdr *)(pkt + sizeof(struct ethhdr) + sizeof(struct iphdr));
    char *http = (char *)(pkt + HTTP_PAYLOAD_OFFSET);

    // Default: valid HTTP packet
    eth->h_proto = ETH_P_IP;
    ip->ihl_version = 0x45;  // IPv4, IHL=5
    ip->protocol = IPPROTO_TCP;
    tcp->dest = HTTP_PORT;

    if (type == 0) {
        // 10%: Non-IP packet (e.g., ARP)
        eth->h_proto = 0x0806;  // ARP
        *pkt_type = 0;
    } else if (type == 1) {
        // 10%: IP but UDP (not TCP)
        ip->protocol = IPPROTO_UDP;
        *pkt_type = 1;
    } else if (type == 2) {
        // 10%: TCP but not HTTP (e.g., port 443)
        tcp->dest = 443;
        *pkt_type = 2;
    } else {
        // 70%: HTTP requests with different paths
        *pkt_type = 3;
        int path_type = pkt_idx % 8;
        const char *paths[] = {
            "GET /api/users HTTP/1.1\r\n",
            "GET /api/orders HTTP/1.1\r\n",
            "GET /api/products HTTP/1.1\r\n",
            "GET /static/style.css HTTP/1.1\r\n",
            "GET /index.html HTTP/1.1\r\n",
            "GET /health HTTP/1.1\r\n",
            "GET /metrics HTTP/1.1\r\n",
            "GET /unknown/path HTTP/1.1\r\n"
        };
        // Copy path (simplified - in real code use proper bounds checking)
        const char *p = paths[path_type];
        for (int i = 0; i < 30 && p[i]; i++) {
            http[i] = p[i];
        }
    }
}

//=============================================================================
// eBPF HOOK - BAD: XDP-style parsing with natural divergence
//=============================================================================

/**
 * This eBPF program parses packets like XDP, causing natural divergence.
 *
 * Traditional eBPF verifier sees:
 *   - All memory accesses are bounds-checked
 *   - All branches are valid and bounded
 *   - No infinite loops
 *
 * GPU reality:
 *   - 10% packets exit at L2 (non-IP)
 *   - 10% packets exit at L3 (non-TCP)
 *   - 10% packets exit at L4 (non-HTTP)
 *   - 70% packets parse HTTP, then diverge on path matching
 *   - Each layer adds more divergence!
 */
__device__ void ebpf_hook_BAD_xdp_parse(unsigned char *pkt_data, int pkt_len) {
    void *data = pkt_data;
    void *data_end = pkt_data + pkt_len;

    // ─────────────────────────────────────────────────────────────────────
    // Layer 2: Parse Ethernet header
    // ─────────────────────────────────────────────────────────────────────
    struct ethhdr *eth = (struct ethhdr *)data;
    if (!bpf_check_bounds(data, data_end, eth, sizeof(*eth))) {
        return;  // Packet too small
    }

    // Check if IPv4
    if (eth->h_proto != ETH_P_IP) {
        // DIVERGENCE: Non-IP packets take this path
        bpf_map_atomic_inc(&protocol_counters[0]);
        return;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Layer 3: Parse IPv4 header
    // ─────────────────────────────────────────────────────────────────────
    struct iphdr *ip = (struct iphdr *)((unsigned char *)eth + sizeof(*eth));
    if (!bpf_check_bounds(data, data_end, ip, sizeof(*ip))) {
        return;
    }

    // Check if TCP
    if (ip->protocol != IPPROTO_TCP) {
        // DIVERGENCE: Non-TCP packets (UDP, ICMP, etc.) take this path
        bpf_map_atomic_inc(&protocol_counters[1]);
        return;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Layer 4: Parse TCP header
    // ─────────────────────────────────────────────────────────────────────
    int ip_hdr_len = (ip->ihl_version & 0x0F) * 4;
    struct tcphdr *tcp = (struct tcphdr *)((unsigned char *)ip + ip_hdr_len);
    if (!bpf_check_bounds(data, data_end, tcp, sizeof(*tcp))) {
        return;
    }

    // Check if HTTP (port 80)
    if (tcp->dest != HTTP_PORT) {
        // DIVERGENCE: Non-HTTP TCP (HTTPS, SSH, etc.) takes this path
        bpf_map_atomic_inc(&protocol_counters[2]);
        return;
    }

    // ─────────────────────────────────────────────────────────────────────
    // Layer 7: Parse HTTP and extract path
    // ─────────────────────────────────────────────────────────────────────
    bpf_map_atomic_inc(&protocol_counters[3]);  // Count HTTP packets

    int tcp_hdr_len = ((tcp->flags >> 12) & 0x0F) * 4;
    if (tcp_hdr_len < 20) tcp_hdr_len = 20;
    char *http_data = (char *)tcp + tcp_hdr_len;

    if (!bpf_check_bounds(data, data_end, http_data, 32)) {
        return;
    }

    // Parse HTTP path - MORE DIVERGENCE based on path content!
    // Each path match is a different branch
    if (http_data[0] == 'G' && http_data[1] == 'E' && http_data[2] == 'T') {
        // GET request - check path
        char *path = http_data + 4;  // Skip "GET "

        if (path[0] == '/' && path[1] == 'a' && path[2] == 'p' && path[3] == 'i') {
            // /api/* routes
            if (path[5] == 'u') {
                bpf_map_atomic_inc(&path_counters[PATH_API_USERS]);
            } else if (path[5] == 'o') {
                bpf_map_atomic_inc(&path_counters[PATH_API_ORDERS]);
            } else if (path[5] == 'p') {
                bpf_map_atomic_inc(&path_counters[PATH_API_PRODUCTS]);
            } else {
                bpf_map_atomic_inc(&path_counters[PATH_OTHER]);
            }
        } else if (path[0] == '/' && path[1] == 's' && path[2] == 't') {
            // /static/*
            bpf_map_atomic_inc(&path_counters[PATH_STATIC]);
        } else if (path[0] == '/' && path[1] == 'i' && path[2] == 'n') {
            // /index.html
            bpf_map_atomic_inc(&path_counters[PATH_INDEX]);
        } else if (path[0] == '/' && path[1] == 'h' && path[2] == 'e') {
            // /health
            bpf_map_atomic_inc(&path_counters[PATH_HEALTH]);
        } else if (path[0] == '/' && path[1] == 'm' && path[2] == 'e') {
            // /metrics
            bpf_map_atomic_inc(&path_counters[PATH_METRICS]);
        } else {
            bpf_map_atomic_inc(&path_counters[PATH_OTHER]);
        }
    }
}

//=============================================================================
// eBPF HOOK - GOOD: Uniform processing (no early returns, batch counters)
//=============================================================================

__device__ unsigned long long per_thread_results[1024 * 1024];

/**
 * GPU-optimized: All threads do same work, store results per-thread
 * No divergence from early returns or path matching
 */
__device__ void ebpf_hook_GOOD_uniform(unsigned char *pkt_data, int pkt_len, int tid) {
    void *data = pkt_data;
    void *data_end = pkt_data + pkt_len;

    // Parse all layers unconditionally (use predication instead of branches)
    struct ethhdr *eth = (struct ethhdr *)data;
    struct iphdr *ip = (struct iphdr *)((unsigned char *)eth + sizeof(*eth));
    struct tcphdr *tcp = (struct tcphdr *)((unsigned char *)ip + 20);

    // Compute result without branching
    int is_ip = (eth->h_proto == ETH_P_IP);
    int is_tcp = (ip->protocol == IPPROTO_TCP);
    int is_http = (tcp->dest == HTTP_PORT);

    // Encode packet type in result (all threads do same operations)
    unsigned long long result = (is_ip << 2) | (is_tcp << 1) | is_http;

    // Store per-thread (reduce later on CPU)
    per_thread_results[tid % (1024 * 1024)] = result;
}

//=============================================================================
// CUDA Kernels
//=============================================================================

__global__ void process_packets_bad(unsigned char *packets, int num_packets, int pkt_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_packets) return;

    unsigned char *my_pkt = packets + tid * pkt_size;

    // Generate packet data (simulating receiving different packets)
    int pkt_type;
    generate_packet(my_pkt, tid, &pkt_type);

    // Run XDP-style eBPF hook - CAUSES DIVERGENCE
    ebpf_hook_BAD_xdp_parse(my_pkt, pkt_size);
}

__global__ void process_packets_good(unsigned char *packets, int num_packets, int pkt_size) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid >= num_packets) return;

    unsigned char *my_pkt = packets + tid * pkt_size;

    // Generate packet data
    int pkt_type;
    generate_packet(my_pkt, tid, &pkt_type);

    // Run uniform eBPF hook - NO DIVERGENCE
    ebpf_hook_GOOD_uniform(my_pkt, pkt_size, tid);
}

__global__ void reset_counters() {
    int tid = threadIdx.x;
    if (tid < MAX_PATHS) path_counters[tid] = 0;
    if (tid < 4) protocol_counters[tid] = 0;
}

//=============================================================================
// Main
//=============================================================================

int main() {
    const int NUM_PACKETS = 1024 * 1024;  // 1M packets
    const int THREADS = 256;
    const int BLOCKS = (NUM_PACKETS + THREADS - 1) / THREADS;
    const int ITERATIONS = 50;

    // Allocate packet buffer
    unsigned char *d_packets;
    cudaMalloc(&d_packets, NUM_PACKETS * PACKET_SIZE);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("╔═══════════════════════════════════════════════════════════════╗\n");
    printf("║  Example 1: Thread Divergence - XDP-style Packet Parsing      ║\n");
    printf("╚═══════════════════════════════════════════════════════════════╝\n\n");

    printf("Simulating XDP-like eBPF program that parses network packets:\n");
    printf("  L2 (Ethernet) → L3 (IPv4) → L4 (TCP) → L7 (HTTP path)\n\n");
    printf("Traffic mix: 10%% non-IP, 10%% non-TCP, 10%% non-HTTP, 70%% HTTP\n");
    printf("HTTP paths: /api/users, /api/orders, /api/products, /static/*,\n");
    printf("            /index.html, /health, /metrics, other\n\n");

    // Warmup
    reset_counters<<<1, 256>>>();
    process_packets_good<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    cudaDeviceSynchronize();

    // Test GOOD (uniform) hook
    reset_counters<<<1, 256>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        process_packets_good<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float good_time;
    cudaEventElapsedTime(&good_time, start, stop);

    // Test BAD (divergent XDP) hook
    reset_counters<<<1, 256>>>();
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    for (int i = 0; i < ITERATIONS; i++) {
        process_packets_bad<<<BLOCKS, THREADS>>>(d_packets, NUM_PACKETS, PACKET_SIZE);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float bad_time;
    cudaEventElapsedTime(&bad_time, start, stop);

    // Copy counters back
    unsigned long long h_path_counters[MAX_PATHS];
    unsigned long long h_protocol_counters[4];
    cudaMemcpyFromSymbol(h_path_counters, path_counters, sizeof(h_path_counters));
    cudaMemcpyFromSymbol(h_protocol_counters, protocol_counters, sizeof(h_protocol_counters));

    printf("Results (%d iterations, %d packets each):\n", ITERATIONS, NUM_PACKETS);
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  GOOD hook (uniform):     %8.2f ms  (%.2f Mpps)\n",
           good_time, (float)NUM_PACKETS * ITERATIONS / good_time / 1000.0f);
    printf("  BAD hook (XDP-style):    %8.2f ms  (%.2f Mpps)\n\n",
           bad_time, (float)NUM_PACKETS * ITERATIONS / bad_time / 1000.0f);

    printf("Performance Impact:\n");
    printf("  BAD vs GOOD: %.2fx slower\n\n", bad_time / good_time);

    printf("Protocol Statistics (last iteration):\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("  Non-IP packets:    %8llu (%.1f%%)\n", h_protocol_counters[0],
           100.0 * h_protocol_counters[0] / NUM_PACKETS);
    printf("  Non-TCP packets:   %8llu (%.1f%%)\n", h_protocol_counters[1],
           100.0 * h_protocol_counters[1] / NUM_PACKETS);
    printf("  Non-HTTP packets:  %8llu (%.1f%%)\n", h_protocol_counters[2],
           100.0 * h_protocol_counters[2] / NUM_PACKETS);
    printf("  HTTP packets:      %8llu (%.1f%%)\n\n", h_protocol_counters[3],
           100.0 * h_protocol_counters[3] / NUM_PACKETS);

    printf("HTTP Path Statistics:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    const char *path_names[] = {
        "/api/users", "/api/orders", "/api/products", "/static/*",
        "/index.html", "/health", "/metrics", "other"
    };
    for (int i = 0; i < MAX_PATHS; i++) {
        printf("  %-15s: %8llu\n", path_names[i], h_path_counters[i]);
    }

    printf("\nAnalysis:\n");
    printf("─────────────────────────────────────────────────────────────────\n");
    printf("The XDP-style parsing causes CASCADING DIVERGENCE:\n\n");
    printf("  Warp of 32 threads processing 32 packets:\n");
    printf("  ├─ 3 packets exit at L2 (non-IP)     → 3 threads idle\n");
    printf("  ├─ 3 packets exit at L3 (non-TCP)    → 3 more threads idle\n");
    printf("  ├─ 3 packets exit at L4 (non-HTTP)   → 3 more threads idle\n");
    printf("  └─ 23 packets parse HTTP             → further diverge on path!\n");
    printf("      ├─ ~3 match /api/users\n");
    printf("      ├─ ~3 match /api/orders\n");
    printf("      ├─ ~3 match /api/products\n");
    printf("      └─ ... (8 different paths = up to 8-way divergence)\n\n");

    printf("Traditional eBPF Verifier:  PASS\n");
    printf("  ✓ All memory accesses bounds-checked\n");
    printf("  ✓ All branches terminate\n");
    printf("  ✓ No infinite loops\n\n");

    printf("GPU-aware Verifier should:  REJECT or WARN\n");
    printf("  ✗ Multiple early returns cause divergence\n");
    printf("  ✗ Nested if-else on packet content\n");
    printf("  ✗ Path matching causes N-way divergence\n");

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_packets);

    return 0;
}
