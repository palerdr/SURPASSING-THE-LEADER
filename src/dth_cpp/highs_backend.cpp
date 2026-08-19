#include "highs_backend.hpp"

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif
#include <Highs.h>
#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace {

using dth::NumericStatus;

[[nodiscard]] NumericStatus translate_status(
    const HighsModelStatus status) noexcept
{
    switch (status) {
    case HighsModelStatus::kOptimal:
        return NumericStatus::Optimal;
    case HighsModelStatus::kInfeasible:
        return NumericStatus::Infeasible;
    case HighsModelStatus::kUnbounded:
        return NumericStatus::Unbounded;
    case HighsModelStatus::kUnboundedOrInfeasible:
        return NumericStatus::InfeasibleOrUnbounded;
    case HighsModelStatus::kIterationLimit:
        return NumericStatus::IterationLimit;
    default:
        return NumericStatus::Failure;
    }
}

void append_coefficient(
    HighsSparseMatrix& matrix,
    const HighsInt row,
    const double value)
{
    if (value == 0.0) {
        return;
    }
    matrix.index_.push_back(row);
    matrix.value_.push_back(value);
}

[[nodiscard]] bool finite_matrix(
    const std::span<const double> matrix) noexcept
{
    return std::ranges::all_of(matrix, [](const double value) {
        return std::isfinite(value);
    });
}

} // namespace

struct dth::HighsBackend::Impl {
    Highs highs{};
    std::string last_error{};

    Impl()
    {
        const auto require_option = [this](
                                        const HighsStatus status,
                                        const char* const name) {
            if (status != HighsStatus::kOk) {
                last_error = std::string{"failed to set HiGHS option: "} + name;
                throw std::runtime_error(last_error);
            }
        };

        require_option(highs.setOptionValue("output_flag", false), "output_flag");
        require_option(highs.setOptionValue("log_to_console", false), "log_to_console");
        require_option(highs.setOptionValue("solver", std::string{"simplex"}), "solver");
        require_option(highs.setOptionValue("simplex_strategy", HighsInt{1}), "simplex_strategy");
        require_option(highs.setOptionValue("parallel", std::string{"off"}), "parallel");
        require_option(highs.setOptionValue("threads", HighsInt{1}), "threads");
        require_option(highs.setOptionValue("random_seed", HighsInt{0}), "random_seed");
        require_option(highs.setOptionValue("presolve", std::string{"off"}), "presolve");
        require_option(
            highs.setOptionValue("primal_feasibility_tolerance", 1e-10),
            "primal_feasibility_tolerance");
        require_option(
            highs.setOptionValue("dual_feasibility_tolerance", 1e-10),
            "dual_feasibility_tolerance");
        require_option(
            highs.setOptionValue("small_matrix_value", 1e-12),
            "small_matrix_value");
        require_option(
            highs.setOptionValue("simplex_iteration_limit", HighsInt{10'000}),
            "simplex_iteration_limit");
    }

    NumericStatus solve(
        HighsModel& model,
        std::span<double> column_values,
        double& objective,
        std::uint64_t& iterations)
    {
        std::ranges::fill(column_values, 0.0);
        objective = 0.0;
        iterations = 0;
        last_error.clear();

        if (highs.clearModel() != HighsStatus::kOk) {
            last_error = "HiGHS failed to clear its previous model";
            return NumericStatus::Failure;
        }

        const HighsStatus pass_status = highs.passModel(model);
        if (pass_status != HighsStatus::kOk) {
            last_error = "HiGHS rejected or modified the supplied model";
            return NumericStatus::Failure;
        }

        const HighsStatus run_status = highs.run();
        const HighsModelStatus model_status = highs.getModelStatus();
        const NumericStatus translated = translate_status(model_status);
        if (translated != NumericStatus::Optimal) {
            last_error = highs.modelStatusToString(model_status);
            return translated;
        }
        if (run_status != HighsStatus::kOk) {
            last_error = "HiGHS returned a warning for an optimal model";
            return NumericStatus::Failure;
        }

        const HighsInfo& info = highs.getInfo();
        const HighsSolution& solution = highs.getSolution();
        if (!solution.value_valid
            || info.primal_solution_status != kSolutionStatusFeasible
            || solution.col_value.size() != column_values.size()
            || !std::isfinite(info.objective_function_value)) {
            last_error = "HiGHS returned an invalid primal solution";
            return NumericStatus::Failure;
        }

        for (std::size_t column = 0; column < column_values.size(); ++column) {
            const double value = solution.col_value[column];
            if (!std::isfinite(value)) {
                last_error = "HiGHS returned a nonfinite column value";
                return NumericStatus::Failure;
            }
            column_values[column] = value;
        }
        objective = info.objective_function_value;
        if (info.simplex_iteration_count < 0) {
            last_error = "HiGHS returned a negative iteration count";
            return NumericStatus::Failure;
        }
        iterations = static_cast<std::uint64_t>(info.simplex_iteration_count);
        return NumericStatus::Optimal;
    }
};

dth::HighsBackend::HighsBackend()
    : impl_(std::make_unique<Impl>())
{
}

dth::HighsBackend::~HighsBackend() = default;
dth::HighsBackend::HighsBackend(HighsBackend&&) noexcept = default;
dth::HighsBackend& dth::HighsBackend::operator=(HighsBackend&&) noexcept = default;

dth::NumericStatus dth::HighsBackend::solve_equalizer(
    const std::span<const double> support_matrix,
    const std::size_t dimension,
    EqualizerRaw& output)
{
    output = {};
    if (dimension == 0 || dimension > kActions
        || support_matrix.size() != dimension * dimension
        || !finite_matrix(support_matrix)) {
        impl_->last_error = "invalid equalizer matrix";
        return NumericStatus::InvalidInput;
    }

    const HighsInt k = static_cast<HighsInt>(dimension);
    const HighsInt variable_count = k + 1;
    const HighsInt row_count = k + 1;

    const auto make_model = [&](const bool transpose) {
        HighsModel model;
        HighsLp& lp = model.lp_;
        lp.num_col_ = variable_count;
        lp.num_row_ = row_count;
        lp.sense_ = ObjSense::kMinimize;
        lp.col_cost_.assign(static_cast<std::size_t>(variable_count), 0.0);
        lp.col_lower_.assign(static_cast<std::size_t>(variable_count), 0.0);
        lp.col_upper_.assign(static_cast<std::size_t>(variable_count), kHighsInf);
        lp.col_lower_[dimension] = -kHighsInf;
        lp.row_lower_.assign(static_cast<std::size_t>(row_count), 0.0);
        lp.row_upper_.assign(static_cast<std::size_t>(row_count), 0.0);
        lp.row_lower_[dimension] = 1.0;
        lp.row_upper_[dimension] = 1.0;

        HighsSparseMatrix& matrix = lp.a_matrix_;
        matrix.format_ = MatrixFormat::kColwise;
        matrix.num_col_ = variable_count;
        matrix.num_row_ = row_count;
        matrix.start_.clear();
        matrix.index_.clear();
        matrix.value_.clear();
        matrix.start_.reserve(static_cast<std::size_t>(variable_count + 1));
        matrix.index_.reserve(dimension * dimension + 2 * dimension);
        matrix.value_.reserve(dimension * dimension + 2 * dimension);

        for (std::size_t column = 0; column < dimension; ++column) {
            matrix.start_.push_back(static_cast<HighsInt>(matrix.value_.size()));
            for (std::size_t row = 0; row < dimension; ++row) {
                const std::size_t source = transpose
                    ? column * dimension + row
                    : row * dimension + column;
                append_coefficient(
                    matrix,
                    static_cast<HighsInt>(row),
                    support_matrix[source]);
            }
            append_coefficient(matrix, k, 1.0);
        }

        matrix.start_.push_back(static_cast<HighsInt>(matrix.value_.size()));
        for (std::size_t row = 0; row < dimension; ++row) {
            append_coefficient(matrix, static_cast<HighsInt>(row), -1.0);
        }
        matrix.start_.push_back(static_cast<HighsInt>(matrix.value_.size()));
        return model;
    };

    std::array<double, kActions + 1> check_solution{};
    std::array<double, kActions + 1> drop_solution{};
    double checker_objective = 0.0;
    double dropper_objective = 0.0;
    std::uint64_t checker_iterations = 0;
    std::uint64_t dropper_iterations = 0;

    HighsModel checker_model = make_model(false);
    NumericStatus status = impl_->solve(
        checker_model,
        std::span<double>{check_solution}.first(dimension + 1),
        checker_objective,
        checker_iterations);
    if (status != NumericStatus::Optimal) {
        return status;
    }

    HighsModel dropper_model = make_model(true);
    status = impl_->solve(
        dropper_model,
        std::span<double>{drop_solution}.first(dimension + 1),
        dropper_objective,
        dropper_iterations);
    if (status != NumericStatus::Optimal) {
        return status;
    }

    std::copy_n(check_solution.begin(), dimension, output.check_mass.begin());
    std::copy_n(drop_solution.begin(), dimension, output.drop_mass.begin());
    output.check_value = check_solution[dimension];
    output.drop_value = drop_solution[dimension];
    output.iterations = checker_iterations + dropper_iterations;
    return NumericStatus::Optimal;
}

dth::NumericStatus dth::HighsBackend::solve_covering(
    const std::span<const double> shifted_matrix,
    const std::size_t dimension,
    CoveringRaw& output)
{
    output = {};
    if (dimension == 0 || dimension > kActions
        || shifted_matrix.size() != dimension * dimension
        || !finite_matrix(shifted_matrix)
        || std::ranges::any_of(shifted_matrix, [](const double value) {
               return value <= 0.0;
           })) {
        impl_->last_error = "invalid shifted covering matrix";
        return NumericStatus::InvalidInput;
    }

    const HighsInt n = static_cast<HighsInt>(dimension);
    const auto make_model = [&](const bool packing) {
        HighsModel model;
        HighsLp& lp = model.lp_;
        lp.num_col_ = n;
        lp.num_row_ = n;
        lp.sense_ = packing ? ObjSense::kMaximize : ObjSense::kMinimize;
        lp.col_cost_.assign(dimension, 1.0);
        lp.col_lower_.assign(dimension, 0.0);
        lp.col_upper_.assign(dimension, kHighsInf);
        lp.row_lower_.assign(dimension, packing ? -kHighsInf : 1.0);
        lp.row_upper_.assign(dimension, packing ? 1.0 : kHighsInf);

        HighsSparseMatrix& matrix = lp.a_matrix_;
        matrix.format_ = MatrixFormat::kColwise;
        matrix.num_col_ = n;
        matrix.num_row_ = n;
        matrix.start_.clear();
        matrix.index_.clear();
        matrix.value_.clear();
        matrix.start_.reserve(dimension + 1);
        matrix.index_.reserve(dimension * dimension);
        matrix.value_.reserve(dimension * dimension);

        for (std::size_t column = 0; column < dimension; ++column) {
            matrix.start_.push_back(static_cast<HighsInt>(matrix.value_.size()));
            for (std::size_t row = 0; row < dimension; ++row) {
                const std::size_t source = packing
                    ? row * dimension + column
                    : column * dimension + row;
                append_coefficient(
                    matrix,
                    static_cast<HighsInt>(row),
                    shifted_matrix[source]);
            }
        }
        matrix.start_.push_back(static_cast<HighsInt>(matrix.value_.size()));
        return model;
    };

    std::array<double, kActions> x{};
    std::array<double, kActions> y{};
    double covering_objective = 0.0;
    double packing_objective = 0.0;
    std::uint64_t covering_iterations = 0;
    std::uint64_t packing_iterations = 0;

    HighsModel covering_model = make_model(false);
    NumericStatus status = impl_->solve(
        covering_model,
        std::span<double>{x}.first(dimension),
        covering_objective,
        covering_iterations);
    if (status != NumericStatus::Optimal) {
        return status;
    }

    HighsModel packing_model = make_model(true);
    status = impl_->solve(
        packing_model,
        std::span<double>{y}.first(dimension),
        packing_objective,
        packing_iterations);
    if (status != NumericStatus::Optimal) {
        return status;
    }

    std::copy_n(x.begin(), dimension, output.x.mass.begin());
    std::copy_n(y.begin(), dimension, output.y.mass.begin());
    output.sum_x = covering_objective;
    output.sum_y = packing_objective;
    output.iterations = covering_iterations + packing_iterations;
    return NumericStatus::Optimal;
}

std::string dth::HighsBackend::version() const
{
    return impl_->highs.version();
}

const std::string& dth::HighsBackend::last_error() const noexcept
{
    return impl_->last_error;
}
