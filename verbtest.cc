#include <atomic>
#include <barrier>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dbg.h>
#include <infiniband/verbs.h>
#include <stddef.h>
#include <sys/mman.h>
#include <thread>
#include <tuple>
#include <vector>

#define rdma_max_rd_atomic 16
#define rdma_ib_port 1
#define rdma_gid_idx 1

const uint64_t cache_line_size = 64;
const uint64_t page_size_4k = 4 << 10;
const uint64_t page_size_2m = 2 << 20;

uint64_t upper_align(uint64_t val, uint64_t align)
{
    return (val + align - 1) & ~(align - 1);
}

std::tuple<void *, size_t> alloc_hugepage(size_t size, uint32_t page_size = 0)
{
    if (size == 0)
        return {nullptr, 0};
    if (page_size == 0)
        page_size = size <= page_size_4k ? page_size_4k : page_size_2m;
    auto aligned_size = upper_align(size, page_size);
    void *res = nullptr;
    if (page_size > page_size_4k)
    {
        res =
            mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
                 MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB | (__builtin_ffsll(page_size) - 1) << MAP_HUGE_SHIFT, 0, 0);
        if (res != MAP_FAILED)
            return {res, aligned_size};
        dbg("try to allocate 4k page", size, aligned_size, page_size);
        aligned_size = upper_align(size, page_size_4k);
    }
    res = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
    if (res == MAP_FAILED) [[unlikely]]
    {
        std::abort();
        return {nullptr, 0};
    }
    return {res, aligned_size};
}

void free_hugepage(void *ptr, size_t size)
{
    if (ptr == nullptr)
        return;
    munmap(ptr, size);
}

int modify_qp_to_init(ibv_qp *qp)
{
    const int flags = IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS;
    struct ibv_qp_attr attr;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_INIT;
    attr.port_num = rdma_ib_port;
    attr.pkey_index = 0;
    attr.qp_access_flags =
        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_ATOMIC;
    if ((rc = ibv_modify_qp(qp, &attr, flags)))
        dbg("failed to modify QP state to INIT");
    return rc;
}

int modify_qp_to_rtr(ibv_qp *qp, uint32_t rqp_num, uint16_t rlid, const ibv_gid *rgid)
{
    const int flags = IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                      IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER;
    struct ibv_qp_attr attr;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTR;
    attr.path_mtu = IBV_MTU_512;
    attr.dest_qp_num = rqp_num;
    attr.rq_psn = 0;
    attr.max_dest_rd_atomic = rdma_max_rd_atomic;
    attr.min_rnr_timer = 12;
    attr.ah_attr.is_global = 0;
    attr.ah_attr.dlid = rlid;
    attr.ah_attr.sl = 0;
    attr.ah_attr.src_path_bits = 0;
    attr.ah_attr.port_num = rdma_ib_port;
    if (rdma_gid_idx >= 0)
    {
        attr.ah_attr.is_global = 1;
        memcpy(&attr.ah_attr.grh.dgid, rgid, sizeof(ibv_gid));
        attr.ah_attr.grh.flow_label = 0;
        attr.ah_attr.grh.hop_limit = 1;
        attr.ah_attr.grh.sgid_index = rdma_gid_idx;
        attr.ah_attr.grh.traffic_class = 0;
    }
    if ((rc = ibv_modify_qp(qp, &attr, flags)))
        dbg("failed to modify QP state to RTR");
    return rc;
}

int modify_qp_to_rts(ibv_qp *qp)
{
    const int flags =
        IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC;
    struct ibv_qp_attr attr;
    int rc;
    memset(&attr, 0, sizeof(attr));
    attr.qp_state = IBV_QPS_RTS;
    attr.timeout = 14;
    attr.retry_cnt = 7;
    attr.rnr_retry = 7;
    attr.sq_psn = 0;
    attr.max_rd_atomic = rdma_max_rd_atomic; // FOR READ
    if ((rc = ibv_modify_qp(qp, &attr, flags)))
        dbg("failed to modify QP state to RTS");
    return rc;
}

static inline uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

inline uint64_t xoshiro256pp(uint64_t s[4])
{
    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];
    s[2] ^= t;
    s[3] = rotl(s[3], 45);
    return result;
}

template <ibv_wr_opcode opcode>
inline void fill_rw_wr(ibv_send_wr *wr, ibv_sge *sge, uint64_t raddr, uint32_t rkey, void *laddr, uint32_t len,
                       uint32_t lkey)
{
    memset(wr, 0, sizeof(ibv_send_wr));
    sge->addr = (uint64_t)laddr;
    sge->length = len;
    sge->lkey = lkey;
    wr->num_sge = 1;
    wr->sg_list = sge;
    wr->opcode = opcode;
    wr->wr.rdma.remote_addr = raddr;
    wr->wr.rdma.rkey = rkey;
}

auto create_ibv_qp(ibv_pd *pd, ibv_cq *cq)
{
    const ibv_qp_cap qp_cap = {
        .max_send_wr = 255,
        .max_recv_wr = 15,
        .max_send_sge = 1,
        .max_recv_sge = 1,
        .max_inline_data = 128,
    };
    ibv_qp *qp;
    struct ibv_qp_init_attr qp_init_attr;
    memset(&qp_init_attr, 0, sizeof(qp_init_attr));
    qp_init_attr.qp_type = IBV_QPT_RC;
    qp_init_attr.sq_sig_all = 0;
    qp_init_attr.send_cq = cq;
    qp_init_attr.recv_cq = cq;
    qp_init_attr.cap = qp_cap;
    qp = ibv_create_qp(pd, &qp_init_attr);
    return qp;
}

ibv_context *ser_ctx, *cli_ctx;
ibv_port_attr ser_port_attr, cli_port_attr;
ibv_pd *ser_pd, *cli_pd;
ibv_cq *ser_cq, *cli_cq;
void *ser_buf, *cli_buf;
size_t ser_buf_size, cli_buf_size;
ibv_mr *ser_mr, *cli_mr;
int cli_num;
int rdma_depth;

auto create_conn_qp(ibv_cq *cq1, ibv_cq *cq2)
{
    auto qp1 = create_ibv_qp(ser_pd, cq1);
    auto qp2 = create_ibv_qp(cli_pd, cq2);
    modify_qp_to_init(qp1);
    modify_qp_to_init(qp2);
    uint32_t rqp_num1 = qp2->qp_num;
    uint32_t rqp_num2 = qp1->qp_num;
    uint16_t rlid1 = cli_port_attr.lid;
    uint16_t rlid2 = ser_port_attr.lid;
    ibv_gid rgid1;
    ibv_gid rgid2;
    ibv_query_gid(cli_ctx, rdma_ib_port, rdma_gid_idx, &rgid1);
    ibv_query_gid(ser_ctx, rdma_ib_port, rdma_gid_idx, &rgid2);
    modify_qp_to_rtr(qp1, rqp_num1, rlid1, &rgid1);
    modify_qp_to_rtr(qp2, rqp_num2, rlid2, &rgid2);
    modify_qp_to_rts(qp1);
    modify_qp_to_rts(qp2);
    return std::make_tuple(qp1, qp2);
}

auto create_ctx_pd()
{
    ibv_context *ib_ctx;
    {
        ibv_device **dev_list = nullptr;
        ibv_device *ib_dev = nullptr;
        int num_devices = 0;

        dev_list = ibv_get_device_list(&num_devices);
        assert(dev_list && num_devices);

        for (int i = 0; i < num_devices; i++)
            if (!strcmp(ibv_get_device_name(dev_list[i]), "mlx5_1"))
            {
                ib_dev = dev_list[i];
                break;
            }
        assert(ib_dev);

        assert(ib_ctx = ibv_open_device(ib_dev));

        ibv_free_device_list(dev_list);
        dev_list = nullptr;
    }

    ibv_port_attr port_attr;
    assert(0 == ibv_query_port(ib_ctx, rdma_ib_port, &port_attr));

    ibv_pd *pd;
    assert(pd = ibv_alloc_pd(ib_ctx));
    return std::make_tuple(ib_ctx, port_attr, pd);
}

std::barrier sync_barrier(2);
std::atomic_bool stop_flag{false};

void ser_main_th()
{
    std::tie(ser_ctx, ser_port_attr, ser_pd) = create_ctx_pd();
    assert(ser_cq = ibv_create_cq(ser_ctx, 3, nullptr, nullptr, 0));
    std::tie(ser_buf, ser_buf_size) = alloc_hugepage(1 << 30);
    ser_mr = ibv_reg_mr(ser_pd, ser_buf, ser_buf_size,
                        IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE |
                            IBV_ACCESS_REMOTE_ATOMIC);
    dbg("server ready");
    sync_barrier.arrive_and_wait();
    while (!stop_flag.load())
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

void cli_main_th()
{
    std::tie(cli_ctx, cli_port_attr, cli_pd) = create_ctx_pd();
    assert(cli_cq = ibv_create_cq(cli_ctx, 3, nullptr, nullptr, 0));
    std::tie(cli_buf, cli_buf_size) = alloc_hugepage(1 << 30);
    cli_mr = ibv_reg_mr(cli_pd, cli_buf, cli_buf_size, IBV_ACCESS_LOCAL_WRITE);
    sync_barrier.arrive_and_wait();
    std::vector<std::tuple<ibv_qp *, ibv_qp *>> fast_qps;
    for (int i = 0; i < 4; ++i)
        fast_qps.emplace_back(create_conn_qp(ser_cq, cli_cq));

    std::barrier cli_barrier(cli_num);
    struct
    {
        uint64_t post_cnt{0};
        uint64_t post_duration{0};
        uint64_t randstate[4];
    } cli_locals[256] __attribute__((aligned(64)));
    for (int i = 0; i < 256; ++i)
    {
        cli_locals[i].randstate[0] = 0x3c16d688 + (i + 1) * 76377679;
        cli_locals[i].randstate[1] = 0xd5ffcf7b + (i + 1) * 33326913;
        cli_locals[i].randstate[2] = 0x36fa1bad + (i + 1) * 48255694;
        cli_locals[i].randstate[3] = 0x2fe8542a + (i + 1) * 73412076;
    }

    auto cli_th = [&](int tid) {
        auto &cli_local = cli_locals[tid];
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(tid + 4, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        ibv_cq *cli_local_cq;
        assert(cli_local_cq = ibv_create_cq(cli_ctx, 127, nullptr, nullptr, 0));
        auto [_, qp] = create_conn_qp(ser_cq, cli_local_cq);
        ibv_send_wr wr[64];
        ibv_sge sge[64];
        uint64_t ser_addr = (uint64_t)ser_buf + tid * upper_align((1 << 30) / cli_num, 4096 * 1024);
        for (int i = 0; i < 64; ++i)
        {
            fill_rw_wr<IBV_WR_RDMA_WRITE>(&wr[i], &sge[i], ser_addr + xoshiro256pp(cli_local.randstate) % 512 * 8,
                                          ser_mr->rkey, (uint8_t *)cli_buf + tid * 4096 * 1024 + i * 128, 128,
                                          cli_mr->lkey);
            wr[i].send_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
            // wr[i].send_flags = IBV_SEND_SIGNALED;
            if (i == 63)
                wr[i].next = nullptr;
            else
                wr[i].next = &wr[i + 1];
        }
        cli_barrier.arrive_and_wait();
        ibv_send_wr *bad_wr;
        ibv_post_send(qp, &wr[64 - rdma_depth], &bad_wr);
        const int check_interval = 100;

        auto ts = std::chrono::high_resolution_clock::now();
        ibv_wc wc[16];
        while (!stop_flag.load())
        {
            for (int t = 0; t < check_interval; ++t)
            {
                int ret = ibv_poll_cq(cli_local_cq, 16, wc);
                if (ret > 0)
                {
                    // for (int i = 0; i < ret; ++i)
                    //   wr[63 - i].wr.rdma.remote_addr =
                    //       ser_addr + xoshiro256pp(cli_local.randstate) % 512 * 8;
                    ibv_post_send(qp, &wr[64 - ret], &bad_wr);
                    cli_local.post_cnt += ret;
                }

                // for (int i = 0; i < rdma_depth; ++i) {

                //   wr[i].wr.rdma.remote_addr = ser_addr + dist(rng) * 8;
                //   ibv_post_send(qp, &wr[i], &bad_wr);
                // }
                // int polled = 0;
                // while (polled < rdma_depth) {
                //   int ret = ibv_poll_cq(cli_local_cq, rdma_depth, wc);
                //   polled += ret;
                //   // if (ret > 0) {
                //   //   for (int i = 0; i < ret; ++i) {
                //   //     if (wc[i].status != IBV_WC_SUCCESS)
                //   //       std::abort();
                //   //     ++polled;
                //   //   }
                //   // } else if (ret < 0)
                //   //   std::abort();
                // }
            }
        }

        auto te = std::chrono::high_resolution_clock::now();
        cli_local.post_duration = (te - ts).count();
    };

    std::vector<std::thread> cli_threads;
    for (int i = 0; i < cli_num; ++i)
        cli_threads.emplace_back(cli_th, i);
    for (auto &th : cli_threads)
        th.join();

    uint64_t total_duration = 0;
    uint64_t post_cnt = 0;
    for (int i = 0; i < cli_num; ++i)
    {
        total_duration += cli_locals[i].post_duration;
        post_cnt += cli_locals[i].post_cnt;
    }
    auto avg_duration = 1. * total_duration / cli_num;
    dbg(post_cnt / avg_duration * 1000);
}

int main(int argc, char **argv)
{
    // ./main [client_num] [rdma_depth]
    // {
    //   char buf[11];
    //   snprintf(buf, 11, "%d", 100);
    //   setenv("MLX5_TOTAL_UUARS", buf, 1);
    //   snprintf(buf, 11, "%d", 4);
    //   setenv("MLX5_NUM_LOW_LAT_UUARS", buf, 1);
    // }
    assert(argc == 3);
    cli_num = atoi(argv[1]);
    rdma_depth = atoi(argv[2]);
    assert(cli_num > 0 && rdma_depth > 0);

    std::jthread ser_th(ser_main_th);
    std::jthread cli_th(cli_main_th);

    // TODO
    std::this_thread::sleep_for(std::chrono::seconds(8));
    stop_flag.store(true);
    return 0;
}