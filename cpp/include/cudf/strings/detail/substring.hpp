#pragma once

#include <cudf/types.hpp>

#include <rmm/cuda_stream_view.hpp>

#include <memory>

namespace cudf {
namespace strings {
namespace detail {

/**
 * @copydoc cudf::json_to_array
 *
 * @param stream CUDA stream used for device memory operations and kernel launches
 */
std::unique_ptr<cudf::column> json_to_array(
  cudf::strings_column_view const& col,
  rmm::cuda_stream_view stream        = rmm::cuda_stream_default,
  rmm::mr::device_memory_resource* mr = rmm::mr::get_current_device_resource());


}  // namespace detail
}  // namespace strings
}  // namespace cudf