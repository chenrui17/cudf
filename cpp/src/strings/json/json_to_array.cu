#include <cudf/column/column_factories.hpp>
#include <cudf/detail/get_value.cuh>
#include <cudf/detail/utilities/cuda.cuh>
#include <cudf/scalar/scalar.hpp>
#include <cudf/scalar/scalar_factories.hpp>
#include <cudf/strings/strings_column_view.hpp>
#include <cudf/types.hpp>
#include <cudf/utilities/error.hpp>

#include <rmm/exec_policy.hpp>

namespace cudf {

class parser {
  protected:
  CUDA_HOST_DEVICE_CALLABLE parser() : input(nullptr), input_len(0), pos(nullptr) {}
  CUDA_HOST_DEVICE_CALLABLE parser(const char* _input, int64_t _input_len)
    : input(_input), input_len(_input_len), pos(_input)
  {
    parse_whitespace();
  }
  CUDA_HOST_DEVICE_CALLABLE bool parse_whitespace()
  {
    while (!eof()) {
      char c = *pos;
      if (c == ' ' || c == '\r' || c == '\n' || c == '\t') {
        pos++;
      } else {
        return true;
      }
    }
    return false;
  }

  CUDA_HOST_DEVICE_CALLABLE bool eof(const char* p) { return p - input >= input_len; }

  CUDA_HOST_DEVICE_CALLABLE bool eof() { return eof(pos); }

  protected:
  char const* input;
  int64_t input_len;
  char const* pos;
};

CUDA_HOST_DEVICE_CALLABLE bool device_strncmp(const char* str1, const char* str2, size_t num_chars)
{
  for (size_t idx = 0; idx < num_chars; idx++) {
    if (str1[idx] != str2[idx]) { return false; }
  }
  return true;
}

struct json_string {
  const char* str;
  int64_t len;

  CUDA_HOST_DEVICE_CALLABLE bool operator==(json_string const& cmp)
  {
    return len == cmp.len && str != nullptr && cmp.str != nullptr &&
           device_strncmp(str, cmp.str, static_cast<size_t>(len));
  }
};

enum first_operator_type { NONE, OBJECT, ARRAY };
struct json_first_operator {
  first_operator_type type;
  json_string name;
  int index;
};

struct json_array_output {
  size_t output_max_len;
  size_t output_len;

  char* first_element;
  char* second_element;

  CUDA_HOST_DEVICE_CALLABLE void add_output(const char* first_str, size_t len_1, const char* second_str, size_t len_2)
  {
    if (first_element != nullptr && second_element != nullptr) {
      // assert output_len + len_1 + len_2 < output_max_len
      memcpy(first_element + output_len, first_str, len_1);
      memcpy(first_element + output_len + len_1, second_str, len_2);
    }
    output_len += len_1;
    output_len += len_2;
  }

  CUDA_HOST_DEVICE_CALLABLE void add_output(json_string str1, json_string str2) { add_output(str1.str, str1.len, str2.str, str2.len); }
};

class json_state : private parser {
 public:
  CUDA_HOST_DEVICE_CALLABLE json_state()
    : parser(), element(first_operator_type::NONE), cur_el_start(nullptr)
  {
  }
  CUDA_HOST_DEVICE_CALLABLE json_state(const char* _input, int64_t _input_len)
    : parser(_input, _input_len), element(first_operator_type::NONE), cur_el_start(nullptr)
  {
  }
  CUDA_HOST_DEVICE_CALLABLE json_first_operator get_first_character()
  {
    char c = parse_char();
    switch (c) {
      case '{': {
        json_first_operator op;
        json_string j_obj{"{", 1};
        op.name = j_obj;
        op.type = OBJECT;
        return op;
      } break;
      case '[': {
        json_first_operator op;
        json_string j_arr{"[", 1};
        op.name = j_arr;
        op.type = ARRAY;
        return op;
      }
      default: break;
    }
    return {NONE};
  }

  CUDA_HOST_DEVICE_CALLABLE bool eof(const char* p) { return p - input >= input_len; }

  CUDA_HOST_DEVICE_CALLABLE bool eof() { return eof(pos); }

  CUDA_HOST_DEVICE_CALLABLE json_string extract_key()
  {
    json_string key;
    parse_string(key, true);
    return key;
  }
  CUDA_HOST_DEVICE_CALLABLE json_string extract_value()
  {
    json_string value;

    return value;
  }
  CUDA_HOST_DEVICE_CALLABLE char parse_char() { return *pos++; }

  first_operator_type element;

  private:
  CUDA_HOST_DEVICE_CALLABLE bool parse_string(json_string& str, bool can_be_empty)
  {
    str.str = nullptr;
    str.len = 0;

    if (parse_whitespace()) {
      if (*pos == '\"') {
        const char* start = ++pos;
        while (!eof()) {
          if (*pos == '\"') {
            str.str = start;
            str.len = pos - start;
            pos++;
            return true;
          }
          pos++;
        }
      }
    }
    return can_be_empty ? true : false;
  }

  const char* cur_el_start;
};

namespace strings {
namespace detail {

namespace {
using namespace cudf;

CUDA_HOST_DEVICE_CALLABLE void parse_json_array(json_state& j_state,
                                                json_array_output& output,
                                                size_t num_rows)
{
  json_first_operator op = j_state.get_first_character();
  switch(op.type) {
    case OBJECT: {

      for (num_rows = 0 ; ; num_rows++)
      {
        json_string key_name;
        json_string key = j_state.extract_key();
        json_string value = j_state.extract_value();
        if (j_state.eof()) break;
        output.add_output(key, value);
      }
    } break;
    case ARRAY: {

    } break;
    case NONE: {

    } break;

    default: break;
  }
}

CUDA_HOST_DEVICE_CALLABLE json_array_output json_to_array_kernel_impl(char const* input,
                                                                      size_t input_len,
                                                                      char* out_buf,
                                                                      size_t out_buf_size,
                                                                      size_t num_rows)
{
  json_array_output output{out_buf_size, 0, out_buf, out_buf};
  json_state j_state(input, input_len);

  parse_json_array(j_state, output, num_rows);
  return output;
}

__global__ void json_to_array_kernel(char const* chars,
                                     size_type const* offsets,
                                     char* out_buf,
                                     size_type* output_offsets,
                                     size_t out_buf_size,
                                     size_t num_rows)
{
  uint64_t const tid = threadIdx.x + (blockDim.x * blockIdx.x);
  json_array_output out = json_to_array_kernel_impl(chars + offsets[tid], // 第tid个数据
                                                    offsets[tid + 1] - offsets[tid],
                                                    out_buf,
                                                    out_buf_size,
                                                    num_rows);

  if (output_offsets != nullptr) { output_offsets[tid] = static_cast<size_type>(out.output_len); }
}

std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                              rmm::cuda_stream_view stream,
                                              rmm::mr::device_memory_resource* mr)
{
  size_t stack_size;
  cudaDeviceGetLimit(&stack_size, cudaLimitStackSize);
  cudaDeviceSetLimit(cudaLimitStackSize, 2048);

  auto offsets = cudf::make_fixed_width_column(
    data_type{type_id::INT32}, col.size() + 1, mask_state::UNALLOCATED, stream, mr);
  cudf::mutable_column_view offsets_view(*offsets);

  cudf::detail::grid_1d const grid{1, col.size()};
  cudf::size_type num_rows = 0;

  // pre-compute for calculate the output column size 
  json_to_array_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      col.chars().head<char>(),
      col.offsets().head<size_type>(),
      nullptr,
      offsets_view.head<size_type>(),
      0,
      num_rows);

  thrust::exclusive_scan(rmm::exec_policy(stream),  // 并行化执行策略
                         offsets_view.head<size_type>(),
                         offsets_view.head<size_type>() + col.size() + 1,
                         offsets_view.head<size_type>(),
                         0);

  size_type output_size = cudf::detail::get_value<size_type>(offsets_view, col.size(), stream);

  std::vector<std::unique_ptr<cudf::column>> col_output;
  
  auto element_col = cudf::make_fixed_width_column(
    data_type{type_id::INT8}, output_size, mask_state::UNALLOCATED, stream, mr);

  col_output.push_back(std::move(element_col));

  cudf::mutable_column_view element_view(*element_col); 

  json_to_array_kernel<<<grid.num_blocks, grid.num_threads_per_block, 0, stream.value()>>>(
      col.chars().head<char>(),
      col.offsets().head<size_type>(),
      element_view.head<char>(),
      nullptr,
      output_size,
      num_rows);

  cudaDeviceSetLimit(cudaLimitStackSize, stack_size);

  return make_structs_column(num_rows,
                             std::move(col_output),
                             0,
                             rmm::device_buffer{},
                             stream,
                             mr);
}


}  // namespace

}  // namespace detail

std::unique_ptr<cudf::column> json_to_array(cudf::strings_column_view const& col,
                                              rmm::mr::device_memory_resource* mr)
{
  return detail::json_to_array(col, 0, mr);
}

}  // namespace strings
}  // namespace cudf 