#include <btllib/bloom_filter.hpp>
#include <btllib/counting_bloom_filter.hpp>
#include <fstream>
#include <omp.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <signal.h>

#include "args.hpp"
#include "bf_utils.hpp"
#include "filter.hpp"
#include "utils.hpp"

template<class BloomFilter>
inline void
save(BloomFilter& bf, const std::string& path)
{
  std::cout << std::endl << "actual output false positive rate: " << bf.get_fpr() << std::endl;
  Timer timer;
  timer.start("[-o] saving to " + path);
  std::locale::global(std::locale::classic());
  bf.save(path);
  timer.stop();
}

int
run(std::vector<std::string> argv)
{
  std::unique_ptr<ProgramArguments> args;
  try {
    args = std::make_unique<ProgramArguments>(argv);
  } catch (const std::runtime_error& err) {
    std::cerr << err.what() << std::endl;
    return EXIT_FAILURE;
  }
  std::cout << args.get()->get_arg_summary();

  omp_set_num_threads(args.get()->num_threads);

  const auto num_elements =
    get_num_elements(args.get()->cmin, args.get()->histogram, args.get()->seeds.size());
  print_stats_table(args.get()->histogram[0], args.get()->histogram[1], num_elements);

  size_t out_size;
  unsigned num_hashes;
  bool out_size_known = args.get()->out_size != 0;
  const auto cascade_fpr = std::cbrt(args.get()->target_err);
  if (out_size_known) {
    out_size = args.get()->out_size;
    num_hashes = (double)out_size / (double)num_elements * log(2);
  } else {
    num_hashes = 3;
    out_size = get_bf_size(num_elements, cascade_fpr, num_hashes);
    out_size *= args.get()->counts ? 8UL : 1UL;
  }

  size_t excludes_size = 0;
  if (args.get()->cmax < 255 && !args.get()->counts) {
    excludes_size = get_bf_size(args.get()->histogram[1] - num_elements, cascade_fpr, num_hashes);
  }

  uint64_t num_bf_elements = args.get()->cmin > 1 ? args.get()->histogram[1] : 0;
  num_bf_elements *= std::max(1UL, args.get()->seeds.size());
  const auto bf_size = get_bf_size(num_bf_elements, cascade_fpr, num_hashes);

  uint64_t num_cbf_elements = 0;
  if (args.get()->cmin == 1 && args.get()->cmax < 255) {
    num_cbf_elements = args.get()->histogram[1];
  } else if (args.get()->cmin > 2) {
    num_cbf_elements = args.get()->histogram[1] - args.get()->histogram[2];
    num_cbf_elements += args.get()->histogram[1] * cascade_fpr;
  }
  num_cbf_elements *= std::max(1UL, args.get()->seeds.size());
  const auto cbf_size = get_bf_size(num_cbf_elements, cascade_fpr, num_hashes);

  size_t ram_usage = bf_size + cbf_size + out_size + excludes_size;

  std::cout << "number of hash functions: " << num_hashes << std::endl;
  std::cout << "cascade false positive rate: " << cascade_fpr << std::endl << std::endl;
  std::cout << "predicted memory usage: " << human_readable(ram_usage) << std::endl;
  std::cout << std::setprecision(3);

  Timer timer;
  const auto read_files = args.get()->reads_paths;
  const auto long_mode = args.get()->long_mode;
  const auto kmer_length = args.get()->kmer_length;
  const auto seeds = args.get()->seeds;
  const auto f0 = args.get()->histogram[0];
  const auto cmin = args.get()->cmin;
  const auto cmax = args.get()->cmax;
  const auto out_path = args.get()->out_path;
  const auto out_size_str = human_readable(out_size);
  const auto excludes_size_str = human_readable(excludes_size);

  if (seeds.empty() && args.get()->counts) {
    timer.start(" allocating output btllib::KmerCountingBloomFilter8 (" + out_size_str + ")");
    btllib::KmerCountingBloomFilter8 out(out_size, num_hashes, kmer_length);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, f0, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (args.get()->seeds.empty()) {
    timer.start(" allocating output btllib::KmerBloomFilter (" + out_size_str + ")");
    btllib::KmerBloomFilter out_include(out_size, num_hashes, kmer_length);
    timer.stop();
    timer.start(" allocating excludes btllib::KmerBloomFilter (" + excludes_size_str + ")");
    btllib::KmerBloomFilter out_exclude(excludes_size + 1, num_hashes, kmer_length);
    timer.stop();
    BloomFilterWrapper out(out_include, out_exclude);
    process(read_files, long_mode, kmer_length, seeds, f0, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (!args.get()->seeds.empty() && args.get()->counts) {
    timer.start(" allocating output btllib::CountingBloomFilter8 (" + out_size_str + ")");
    btllib::CountingBloomFilter8 out(out_size, num_hashes);
    timer.stop();
    process(read_files, long_mode, kmer_length, seeds, f0, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  } else if (!args.get()->seeds.empty()) {
    timer.start(" allocating output btllib::SeedBloomFilter (" + out_size_str + ")");
    btllib::SeedBloomFilter out_include(out_size, kmer_length, seeds, num_hashes);
    timer.stop();
    timer.start(" allocating excludes btllib::SeedBloomFilter (" + excludes_size_str + ")");
    btllib::SeedBloomFilter out_exclude(excludes_size + 1, kmer_length, seeds, num_hashes);
    timer.stop();
    BloomFilterWrapper out(out_include, out_exclude);
    process(read_files, long_mode, kmer_length, seeds, f0, cmin, cmax, bf_size, cbf_size, out);
    save(out, out_path);
  }

  return EXIT_SUCCESS;
}

PYBIND11_MODULE(filter, m)
{
  m.def("run", &run);
}