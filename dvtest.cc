#include <cassert>
#include <infiniband/mlx5dv.h>
#include <infiniband/mlx5_user_ioctl_verbs.h>

#include <sys/mman.h>
#include <tuple>

#include <dbg.h>

const uint64_t cache_line_size = 64;
const uint64_t page_size_4k = 4 << 10;
const uint64_t page_size_2m = 2 << 20;

uint64_t upper_align(uint64_t val, uint64_t align) {
  return (val + align - 1) & ~(align - 1);
}

std::tuple<void *, size_t> alloc_hugepage(size_t size, uint32_t page_size = 0) {
  if (size == 0)
    return {nullptr, 0};
  if (page_size == 0)
    page_size = size <= page_size_4k ? page_size_4k : page_size_2m;
  auto aligned_size = upper_align(size, page_size);
  void *res = nullptr;
  if (page_size > page_size_4k) {
    res = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
               MAP_ANONYMOUS | MAP_PRIVATE | MAP_HUGETLB |
                   (__builtin_ffsll(page_size) - 1) << MAP_HUGE_SHIFT,
               0, 0);
    if (res != MAP_FAILED)
      return {res, aligned_size};
    dbg("try to allocate 4k page", size, aligned_size, page_size);
    aligned_size = upper_align(size, page_size_4k);
  }
  res = mmap(nullptr, aligned_size, PROT_READ | PROT_WRITE,
             MAP_ANONYMOUS | MAP_PRIVATE, 0, 0);
  if (res == MAP_FAILED) [[unlikely]] {
    std::abort();
    return {nullptr, 0};
  }
  return {res, aligned_size};
}


auto create_ctx_pd() {
  ibv_context *ib_ctx;
  {
    ibv_device **dev_list = nullptr;
    ibv_device *ib_dev = nullptr;
    int num_devices = 0;

    dev_list = ibv_get_device_list(&num_devices);
    assert(dev_list && num_devices);

    for (int i = 0; i < num_devices; i++)
      if (!strcmp(ibv_get_device_name(dev_list[i]), "mlx5_0")) {
        ib_dev = dev_list[i];
        break;
      }
    assert(ib_dev);

    assert(ib_ctx = ibv_open_device(ib_dev));

    ibv_free_device_list(dev_list);
    dev_list = nullptr;
  }

  ibv_port_attr port_attr;
  assert(0 == ibv_query_port(ib_ctx, 1, &port_attr));

  ibv_pd *pd;
  assert(pd = ibv_alloc_pd(ib_ctx));
  return std::make_tuple(ib_ctx, port_attr, pd);
}

int main(){
  return 0;
}