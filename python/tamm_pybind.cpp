#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
// #include <cstdlib>
#include <limits>
#include <memory>
// #include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "tamm/label_translator.hpp"
#include "tamm/tamm.hpp"

namespace py = pybind11;

namespace {

struct RuntimeState {
  bool initialized  = false;
  bool context_live = false;
};

RuntimeState& runtime_state() {
  static RuntimeState state;
  return state;
}

/*
std::optional<int> getenv_positive_int(const char* name) {
  const char* raw = std::getenv(name);
  if(raw == nullptr || raw[0] == '\0') return std::nullopt;

  try {
    const int value = std::stoi(raw);
    if(value > 0) return value;
  } catch(...) {
  }
  return std::nullopt;
}

int detect_mpi_world_size_guess() {
  const char* env_vars[] = {"OMPI_COMM_WORLD_SIZE", "PMI_SIZE", "MV2_COMM_WORLD_SIZE",
                            "SLURM_NTASKS"};
  for(const auto* name: env_vars) {
    if(const auto value = getenv_positive_int(name)) return *value;
  }
  return 1;
}

std::optional<int> detect_mpi_local_size_guess() {
  const char* env_vars[] = {"OMPI_COMM_WORLD_LOCAL_SIZE", "MPI_LOCALNRANKS",
                            "MV2_COMM_WORLD_LOCAL_SIZE"};
  for(const auto* name: env_vars) {
    if(const auto value = getenv_positive_int(name)) return value;
  }
  return std::nullopt;
}

int detect_progress_ranks_per_node() {
  const auto env_value = getenv_positive_int("GA_NUM_PROGRESS_RANKS_PER_NODE");
  if(env_value && *env_value < 16) return *env_value;
  return 1;
}

void precheck_ga_progress_rank_requirement() {
  const int required_per_node = 2 * detect_progress_ranks_per_node();
  const auto local_size       = detect_mpi_local_size_guess();
  const int  world_size       = detect_mpi_world_size_guess();

  if(local_size.has_value()) {
    if(*local_size < required_per_node) {
      throw std::runtime_error(
        "This TAMM/GA build requires at least " + std::to_string(required_per_node) +
        " MPI ranks per node. Launch with mpirun, e.g. `mpirun -np 2 python "
        "tensor_contraction.py`.");
    }
    return;
  }

  // Fallback for environments without *_LOCAL_SIZE metadata.
  if(world_size < required_per_node) {
    throw std::runtime_error(
      "This TAMM/GA build requires at least " + std::to_string(required_per_node) +
      " MPI ranks per node. Launch with mpirun, e.g. `mpirun -np 2 python "
      "tensor_contraction.py`.");
  }
}
*/

tamm::Index to_index(std::size_t value, const std::string& what) {
  if(value > static_cast<std::size_t>(std::numeric_limits<tamm::Index>::max())) {
    throw std::out_of_range(what + " is too large for TAMM Index");
  }
  return static_cast<tamm::Index>(value);
}

tamm::Tile to_tile(std::size_t value, const std::string& what) {
  if(value == 0) throw std::invalid_argument(what + " must be > 0");
  if(value > static_cast<std::size_t>(std::numeric_limits<tamm::Tile>::max())) {
    throw std::out_of_range(what + " is too large for TAMM Tile");
  }
  return static_cast<tamm::Tile>(value);
}

std::vector<std::size_t> tensor_shape(const tamm::Tensor<double>& tensor) {
  std::vector<std::size_t> dims;
  dims.reserve(tensor.num_modes());
  for(const auto& tis: tensor.tiled_index_spaces()) {
    dims.push_back(tis.index_space().num_indices());
  }
  return dims;
}

std::vector<std::size_t> contiguous_strides(const std::vector<std::size_t>& dims) {
  std::vector<std::size_t> strides(dims.size(), 1);
  if(dims.empty()) return strides;
  for(std::size_t i = dims.size() - 1; i > 0; --i) { strides[i - 1] = strides[i] * dims[i]; }
  return strides;
}

std::size_t linear_to_global_flat(std::size_t linear, const std::vector<std::size_t>& block_dims,
                                  const std::vector<std::size_t>& block_offsets,
                                  const std::vector<std::size_t>& global_strides) {
  if(block_dims.empty()) return 0;

  std::size_t flat = 0;
  for(std::size_t axis = block_dims.size(); axis-- > 0;) {
    const auto local_idx = linear % block_dims[axis];
    linear /= block_dims[axis];
    flat += (block_offsets[axis] + local_idx) * global_strides[axis];
  }
  return flat;
}

tamm::LabeledTensor<double> make_labeled_tensor(const tamm::Tensor<double>&            tensor,
                                                const std::vector<std::string>& labels) {
  if(labels.size() != tensor.num_modes()) {
    throw std::invalid_argument("Label count must match tensor rank.");
  }

  const auto& tis_vec = tensor.tiled_index_spaces();
  tamm::IndexLabelVec ilv;
  ilv.reserve(labels.size());

  for(std::size_t i = 0; i < labels.size(); ++i) {
    if(labels[i].empty()) throw std::invalid_argument("Labels cannot be empty.");
    ilv.push_back(tis_vec[i].string_label(labels[i]));
  }

  return tamm::LabeledTensor<double>{tensor, ilv};
}

std::vector<std::string> parse_index_term(const std::string& term) {
  std::vector<std::string> labels;
  labels.reserve(term.size());

  for(const char ch: term) {
    if(std::isspace(static_cast<unsigned char>(ch))) continue;
    if(ch == '.') {
      throw std::invalid_argument("Ellipsis is not supported in contract_einsum.");
    }
    if(std::isalnum(static_cast<unsigned char>(ch)) || ch == '_') {
      labels.emplace_back(1, ch);
    }
    else {
      throw std::invalid_argument("Unexpected character in einsum term.");
    }
  }
  return labels;
}

std::tuple<std::vector<std::string>, std::vector<std::string>, std::vector<std::string>>
parse_einsum_expression(const std::string& expression) {
  std::string expr;
  expr.reserve(expression.size());
  for(const char ch: expression) {
    if(!std::isspace(static_cast<unsigned char>(ch))) expr.push_back(ch);
  }

  const auto arrow_pos = expr.find("->");
  if(arrow_pos == std::string::npos) {
    throw std::invalid_argument("Einsum expression must contain '->'.");
  }

  const auto comma_pos = expr.find(',');
  if(comma_pos == std::string::npos || comma_pos > arrow_pos) {
    throw std::invalid_argument("Einsum expression must be of form 'ab,bc->ac'.");
  }
  if(expr.find(',', comma_pos + 1) != std::string::npos &&
     expr.find(',', comma_pos + 1) < arrow_pos) {
    throw std::invalid_argument("Only binary einsum expressions are supported.");
  }

  const auto lhs_a = expr.substr(0, comma_pos);
  const auto lhs_b = expr.substr(comma_pos + 1, arrow_pos - comma_pos - 1);
  const auto rhs   = expr.substr(arrow_pos + 2);

  return {parse_index_term(lhs_a), parse_index_term(lhs_b), parse_index_term(rhs)};
}

std::size_t choose_auto_tile_size(std::size_t size) {
  if(size == 0) throw std::invalid_argument("size must be > 0");

  const std::size_t cap = std::min<std::size_t>(size, 64);
  for(std::size_t t = cap; t > 1; --t) {
    if(size % t == 0) return t;
  }
  return 1;
}

std::vector<std::size_t> parse_shape(py::handle shape_obj) {
  if(py::isinstance<py::int_>(shape_obj)) {
    return {shape_obj.cast<std::size_t>()};
  }

  if(!py::isinstance<py::sequence>(shape_obj)) {
    throw std::invalid_argument("shape must be an int or a sequence of ints.");
  }

  std::vector<std::size_t> shape;
  for(const auto& dim: py::reinterpret_borrow<py::sequence>(shape_obj)) {
    shape.push_back(py::cast<std::size_t>(dim));
  }
  if(shape.empty()) throw std::invalid_argument("shape cannot be empty.");
  return shape;
}

std::vector<std::size_t> parse_tiles(py::handle tile_obj, const std::vector<std::size_t>& shape) {
  if(tile_obj.is_none()) {
    std::vector<std::size_t> tiles;
    tiles.reserve(shape.size());
    for(const auto dim: shape) tiles.push_back(choose_auto_tile_size(dim));
    return tiles;
  }

  if(py::isinstance<py::int_>(tile_obj)) {
    const auto t = py::cast<std::size_t>(tile_obj);
    if(t == 0) throw std::invalid_argument("tile must be > 0.");
    return std::vector<std::size_t>(shape.size(), t);
  }

  if(!py::isinstance<py::sequence>(tile_obj)) {
    throw std::invalid_argument("tile must be None, an int, or a sequence of ints.");
  }

  std::vector<std::size_t> tiles;
  for(const auto& t: py::reinterpret_borrow<py::sequence>(tile_obj)) {
    const auto tv = py::cast<std::size_t>(t);
    if(tv == 0) throw std::invalid_argument("tile values must be > 0.");
    tiles.push_back(tv);
  }
  if(tiles.size() != shape.size()) {
    throw std::invalid_argument("tile sequence length must match shape rank.");
  }
  return tiles;
}

bool same_index_space(const tamm::TiledIndexSpace& lhs, const tamm::TiledIndexSpace& rhs) {
  return lhs.index_space().num_indices() == rhs.index_space().num_indices() &&
         lhs.input_tile_size() == rhs.input_tile_size();
}

class PyTiledIndexSpace {
public:
  explicit PyTiledIndexSpace(tamm::TiledIndexSpace tis): tis_{std::move(tis)} {}

  const tamm::TiledIndexSpace& value() const { return tis_; }

  std::size_t size() const { return tis_.index_space().num_indices(); }

  std::size_t tile_size() const { return tis_.input_tile_size(); }

private:
  tamm::TiledIndexSpace tis_;
};

class PyTensor {
public:
  PyTensor(): tensor_{} {}
  explicit PyTensor(const std::vector<tamm::TiledIndexSpace>& spaces): tensor_{spaces} {}

  tamm::Tensor<double>& value() { return tensor_; }
  const tamm::Tensor<double>& value() const { return tensor_; }

  std::size_t rank() const { return tensor_.num_modes(); }

  std::vector<std::size_t> shape() const { return tensor_shape(tensor_); }

  bool is_allocated() const { return tensor_.is_allocated(); }

private:
  tamm::Tensor<double> tensor_;
};

class PyTammContext {
public:
  PyTammContext(tamm::DistributionKind distribution_kind = tamm::DistributionKind::nw,
                tamm::MemoryManagerKind memory_manager_kind = tamm::MemoryManagerKind::ga,
                // tamm::MemoryManagerKind memory_manager_kind = tamm::MemoryManagerKind::local,
                bool finalize_mpi = true):
    finalize_mpi_{finalize_mpi} {
    auto& state = runtime_state();
    if(state.context_live) {
      throw std::runtime_error(
        "Only one active tamm.Context is supported at a time in this wrapper.");
    }
    if(!state.initialized) {
      // precheck_ga_progress_rank_requirement();
      int   argc = 1;
      char  app_name[] = "pytamm";
      char* argv[]     = {app_name, nullptr};
      tamm::initialize(argc, argv);
      state.initialized = true;
    }

    state.context_live = true;
    pg_                = tamm::ProcGroup::create_world_coll();
    ec_ = std::make_unique<tamm::ExecutionContext>(pg_, distribution_kind, memory_manager_kind);
  }

  ~PyTammContext() {
    try {
      close();
    } catch(...) {
    }
  }

  void close() {
    if(closed_) return;

    ec_.reset();

    auto& state      = runtime_state();
    state.context_live = false;
    if(state.initialized) {
      tamm::finalize(finalize_mpi_);
      state.initialized = false;
    }
    closed_ = true;
  }

  int rank() const { return static_cast<int>(ec().pg().rank().value()); }

  int size() const { return static_cast<int>(ec().pg().size().value()); }

  void barrier() const { ec().pg().barrier(); }

  PyTiledIndexSpace tiled_index_space(std::size_t size, std::size_t tile_size = 0) const {
    const auto effective_tile =
      (tile_size == 0) ? choose_auto_tile_size(size) : tile_size;
    const auto is             = tamm::IndexSpace{tamm::range(to_index(size, "size"))};
    return PyTiledIndexSpace{
      tamm::TiledIndexSpace{is, to_tile(effective_tile, "tile_size")}
    };
  }

  PyTensor tensor(const std::vector<PyTiledIndexSpace>& spaces) const {
    std::vector<tamm::TiledIndexSpace> tis_vec;
    tis_vec.reserve(spaces.size());
    for(const auto& space: spaces) tis_vec.push_back(space.value());
    return PyTensor{tis_vec};
  }

  void allocate(py::args tensors) const {
    if(tensors.size() == 0) return;

    tamm::Scheduler sch{ec()};
    for(const auto& obj: tensors) {
      auto& tensor = py::cast<PyTensor&>(obj);
      sch.allocate(tensor.value());
    }
    sch.execute();
  }

  void deallocate(py::args tensors) const {
    if(tensors.size() == 0) return;

    tamm::Scheduler sch{ec()};
    for(const auto& obj: tensors) {
      auto& tensor = py::cast<PyTensor&>(obj);
      sch.deallocate(tensor.value());
    }
    sch.execute();
  }

  void fill(PyTensor& tensor, double value) const {
    tamm::Scheduler sch{ec()};
    sch(tensor.value()() = value).execute();
  }

  double norm(const PyTensor& tensor) const { return tamm::norm(ec(), tensor.value()); }

  void contract(PyTensor& out, const std::vector<std::string>& out_labels, const PyTensor& lhs,
                const std::vector<std::string>& lhs_labels, const PyTensor& rhs,
                const std::vector<std::string>& rhs_labels, double alpha = 1.0,
                bool add = false) const {
    auto lt_out = make_labeled_tensor(out.value(), out_labels);
    auto lt_lhs = make_labeled_tensor(lhs.value(), lhs_labels);
    auto lt_rhs = make_labeled_tensor(rhs.value(), rhs_labels);

    tamm::Scheduler sch{ec()};
    if(add) sch(lt_out += alpha * lt_lhs * lt_rhs).execute();
    else sch(lt_out = alpha * lt_lhs * lt_rhs).execute();
  }

  void contract_einsum(PyTensor& out, const PyTensor& lhs, const PyTensor& rhs,
                       const std::string& expression, double alpha = 1.0,
                       bool add = false) const {
    auto [lhs_labels, rhs_labels, out_labels] = parse_einsum_expression(expression);
    contract(out, out_labels, lhs, lhs_labels, rhs, rhs_labels, alpha, add);
  }

  PyTensor einsum(const std::string& expression, const PyTensor& lhs, const PyTensor& rhs,
                  py::object out = py::none(), double alpha = 1.0, bool add = false) const {
    if(out.is_none()) {
      auto out_tensor = tensor_from_einsum(expression, lhs, rhs);
      allocate_one(out_tensor);
      contract_einsum(out_tensor, lhs, rhs, expression, alpha, add);
      return out_tensor;
    }

    auto& out_tensor = py::cast<PyTensor&>(out);
    if(!out_tensor.is_allocated()) allocate_one(out_tensor);
    contract_einsum(out_tensor, lhs, rhs, expression, alpha, add);
    return out_tensor;
  }

  PyTensor zeros(py::object shape, py::object tile = py::none()) const {
    auto tensor = tensor_from_shape(shape, tile);
    allocate_one(tensor);
    fill(tensor, 0.0);
    return tensor;
  }

  PyTensor ones(py::object shape, py::object tile = py::none()) const {
    auto tensor = tensor_from_shape(shape, tile);
    allocate_one(tensor);
    fill(tensor, 1.0);
    return tensor;
  }

  py::array_t<double> to_numpy(const PyTensor& tensor) const {
    if(!tensor.value().is_allocated()) {
      throw std::runtime_error("Tensor must be allocated before calling to_numpy.");
    }

    const auto dims = tensor_shape(tensor.value());
    std::vector<py::ssize_t> py_dims;
    py_dims.reserve(dims.size());
    for(const auto d: dims) py_dims.push_back(static_cast<py::ssize_t>(d));

    py::array_t<double> out(py_dims);
    auto* out_ptr = out.mutable_data();

    const auto strides = contiguous_strides(dims);
    auto       lt      = tensor.value()();

    for(auto it: tensor.value().loop_nest()) {
      const auto block_id = tamm::internal::translate_blockid(it, lt);
      if(!tensor.value().is_non_zero(block_id)) continue;

      const auto block_size    = tensor.value().block_size(block_id);
      const auto block_dims    = tensor.value().block_dims(block_id);
      const auto block_offsets = tensor.value().block_offsets(block_id);

      std::vector<double> buffer(block_size, 0.0);
      tensor.value().get(block_id, buffer);

      for(std::size_t linear = 0; linear < block_size; ++linear) {
        const auto flat = linear_to_global_flat(linear, block_dims, block_offsets, strides);
        out_ptr[flat]   = buffer[linear];
      }
    }

    return out;
  }

  void from_numpy(
    PyTensor& tensor,
    const py::array_t<double, py::array::c_style | py::array::forcecast>& array) const {
    if(!tensor.value().is_allocated()) {
      throw std::runtime_error("Tensor must be allocated before calling from_numpy.");
    }

    const auto dims = tensor_shape(tensor.value());
    if(static_cast<std::size_t>(array.ndim()) != dims.size()) {
      throw std::invalid_argument("Input NumPy rank does not match tensor rank.");
    }
    for(std::size_t i = 0; i < dims.size(); ++i) {
      if(static_cast<std::size_t>(array.shape(i)) != dims[i]) {
        throw std::invalid_argument("Input NumPy shape does not match tensor shape.");
      }
    }

    const auto* in_ptr  = array.data();
    const auto  strides = contiguous_strides(dims);
    auto        lt      = tensor.value()();

    for(auto it: tensor.value().loop_nest()) {
      const auto block_id = tamm::internal::translate_blockid(it, lt);
      if(!tensor.value().is_non_zero(block_id)) continue;

      const auto block_size    = tensor.value().block_size(block_id);
      const auto block_dims    = tensor.value().block_dims(block_id);
      const auto block_offsets = tensor.value().block_offsets(block_id);

      std::vector<double> buffer(block_size, 0.0);
      for(std::size_t linear = 0; linear < block_size; ++linear) {
        const auto flat = linear_to_global_flat(linear, block_dims, block_offsets, strides);
        buffer[linear]  = in_ptr[flat];
      }
      tensor.value().put(block_id, buffer);
    }
  }

private:
  void allocate_one(PyTensor& tensor) const {
    tamm::Scheduler sch{ec()};
    sch.allocate(tensor.value()).execute();
  }

  PyTensor tensor_from_shape(py::handle shape_obj, py::handle tile_obj) const {
    if(py::isinstance<py::sequence>(shape_obj)) {
      auto seq = py::reinterpret_borrow<py::sequence>(shape_obj);
      if(py::len(seq) > 0 && py::isinstance<PyTiledIndexSpace>(seq[0])) {
        if(!tile_obj.is_none()) {
          throw std::invalid_argument(
            "tile must be None when shape is a list of TiledIndexSpace objects.");
        }
        std::vector<tamm::TiledIndexSpace> tis_vec;
        tis_vec.reserve(py::len(seq));
        for(const auto& item: seq) {
          const auto& tis = py::cast<PyTiledIndexSpace&>(item);
          tis_vec.push_back(tis.value());
        }
        return PyTensor{tis_vec};
      }
    }

    const auto shape = parse_shape(shape_obj);
    const auto tiles = parse_tiles(tile_obj, shape);

    std::vector<tamm::TiledIndexSpace> tis_vec;
    tis_vec.reserve(shape.size());
    for(std::size_t i = 0; i < shape.size(); ++i) {
      auto tis = tiled_index_space(shape[i], tiles[i]);
      tis_vec.push_back(tis.value());
    }
    return PyTensor{tis_vec};
  }

  PyTensor tensor_from_einsum(const std::string& expression, const PyTensor& lhs,
                              const PyTensor& rhs) const {
    auto [lhs_labels, rhs_labels, out_labels] = parse_einsum_expression(expression);
    const auto& lhs_spaces                    = lhs.value().tiled_index_spaces();
    const auto& rhs_spaces                    = rhs.value().tiled_index_spaces();

    if(lhs_labels.size() != lhs_spaces.size() || rhs_labels.size() != rhs_spaces.size()) {
      throw std::invalid_argument("Einsum expression rank does not match operand rank.");
    }

    std::unordered_map<std::string, tamm::TiledIndexSpace> label_to_space;
    for(std::size_t i = 0; i < lhs_labels.size(); ++i) {
      label_to_space.emplace(lhs_labels[i], lhs_spaces[i]);
    }

    for(std::size_t i = 0; i < rhs_labels.size(); ++i) {
      const auto& label = rhs_labels[i];
      auto        it    = label_to_space.find(label);
      if(it == label_to_space.end()) {
        label_to_space.emplace(label, rhs_spaces[i]);
        continue;
      }
      if(!same_index_space(it->second, rhs_spaces[i])) {
        throw std::invalid_argument("Mismatched tiled index space for label '" + label + "'.");
      }
    }

    std::vector<tamm::TiledIndexSpace> out_spaces;
    out_spaces.reserve(out_labels.size());
    for(const auto& label: out_labels) {
      auto it = label_to_space.find(label);
      if(it == label_to_space.end()) {
        throw std::invalid_argument("Output label '" + label + "' missing from inputs.");
      }
      out_spaces.push_back(it->second);
    }
    return PyTensor{out_spaces};
  }

  tamm::ExecutionContext& ec() const {
    if(closed_ || ec_ == nullptr) throw std::runtime_error("Context is closed.");
    return *ec_;
  }

  tamm::ProcGroup                          pg_;
  std::unique_ptr<tamm::ExecutionContext> ec_;
  bool                                     finalize_mpi_ = true;
  bool                                     closed_       = false;
};

} // namespace

PYBIND11_MODULE(pytamm, m) {
  m.doc() = "Core TAMM tensor wrapper implemented with pybind11";

  py::enum_<tamm::DistributionKind>(m, "DistributionKind")
    .value("NW", tamm::DistributionKind::nw)
    .value("DENSE", tamm::DistributionKind::dense)
    .value("SIMPLE_ROUND_ROBIN", tamm::DistributionKind::simple_round_robin)
    .export_values();

  py::enum_<tamm::MemoryManagerKind>(m, "MemoryManagerKind")
    .value("GA", tamm::MemoryManagerKind::ga)
    .value("LOCAL", tamm::MemoryManagerKind::local)
    .export_values();

  py::class_<PyTiledIndexSpace>(m, "TiledIndexSpace")
    .def_property_readonly("size", &PyTiledIndexSpace::size)
    .def_property_readonly("tile_size", &PyTiledIndexSpace::tile_size);

  py::class_<PyTensor>(m, "Tensor")
    .def_property_readonly("shape", &PyTensor::shape)
    .def_property_readonly("rank", &PyTensor::rank)
    .def("is_allocated", &PyTensor::is_allocated);

  py::class_<PyTammContext>(m, "Context")
    .def(py::init<tamm::DistributionKind, tamm::MemoryManagerKind, bool>(),
         py::arg("distribution")  = tamm::DistributionKind::nw,
         py::arg("memory_manager") = tamm::MemoryManagerKind::ga,
         py::arg("finalize_mpi")  = true)
    .def("__enter__", [](PyTammContext& self) -> PyTammContext& { return self; },
         py::return_value_policy::reference_internal)
    .def("__exit__",
         [](PyTammContext& self, const py::object&, const py::object&, const py::object&) {
           self.close();
         })
    .def("close", &PyTammContext::close, "Finalize TAMM runtime for this context.")
    .def("rank", &PyTammContext::rank)
    .def("size", &PyTammContext::size)
    .def("barrier", &PyTammContext::barrier)
    .def("tiled_index_space", &PyTammContext::tiled_index_space, py::arg("size"),
         py::arg("tile_size") = 0)
    .def("tensor", &PyTammContext::tensor, py::arg("spaces"))
    .def("allocate", &PyTammContext::allocate, "Allocate one or more tensors.")
    .def("deallocate", &PyTammContext::deallocate, "Deallocate one or more tensors.")
    .def("fill", &PyTammContext::fill, py::arg("tensor"), py::arg("value"))
    .def("norm", &PyTammContext::norm, py::arg("tensor"))
    .def("contract", &PyTammContext::contract, py::arg("out"), py::arg("out_labels"),
         py::arg("lhs"), py::arg("lhs_labels"), py::arg("rhs"), py::arg("rhs_labels"),
         py::arg("alpha") = 1.0, py::arg("add") = false)
    .def("contract_einsum", &PyTammContext::contract_einsum, py::arg("out"), py::arg("lhs"),
         py::arg("rhs"), py::arg("expression"), py::arg("alpha") = 1.0,
         py::arg("add") = false)
    .def("einsum", &PyTammContext::einsum, py::arg("expression"), py::arg("lhs"),
         py::arg("rhs"), py::arg("out") = py::none(), py::arg("alpha") = 1.0,
         py::arg("add") = false)
    .def("zeros", &PyTammContext::zeros, py::arg("shape"), py::arg("tile") = py::none())
    .def("ones", &PyTammContext::ones, py::arg("shape"), py::arg("tile") = py::none())
    .def("to_numpy", &PyTammContext::to_numpy, py::arg("tensor"))
    .def("from_numpy", &PyTammContext::from_numpy, py::arg("tensor"), py::arg("array"));

  // Alias to mirror numpy-like import/use style.
  m.attr("TammContext") = m.attr("Context");

  m.def("zeros", [](PyTammContext& ctx, py::object shape, py::object tile) {
    return ctx.zeros(shape, tile);
  }, py::arg("ctx"), py::arg("shape"), py::arg("tile") = py::none());

  m.def("ones", [](PyTammContext& ctx, py::object shape, py::object tile) {
    return ctx.ones(shape, tile);
  }, py::arg("ctx"), py::arg("shape"), py::arg("tile") = py::none());
}
