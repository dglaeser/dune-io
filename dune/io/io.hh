#ifndef Udune_io_HH
#define Udune_io_HH

#include <config.h>

#include <type_traits>

#include <gridformat/gridformat.hpp>
#include <gridformat/traits/dune.hpp>

#if HAVE_DUNE_LOCALFUNCTIONS
inline constexpr bool __have_dune_local_functions = true;
#else
inline constexpr bool __have_dune_local_functions = false;
#endif

#if HAVE_MPI
#include <mpi.h>
using __MPI_COMM = MPI_Comm;
inline constexpr bool __have_mpi = true;
#else
struct __MPI_COMM {};
inline constexpr bool __have_mpi = false;
#endif

namespace Dune::IO {

namespace VTK { using namespace GridFormat::VTK; }
namespace Format { using namespace GridFormat::Formats; }
namespace Encoding { using namespace GridFormat::Encoding; }
namespace Compression { using namespace GridFormat::Compression; using GridFormat::none; }
namespace Precision {
    using GridFormat::float32;
    using GridFormat::float64;

    using GridFormat::uint64;
    using GridFormat::uint32;
    using GridFormat::uint16;
    using GridFormat::uint8;

    using GridFormat::int64;
    using GridFormat::int32;
    using GridFormat::int16;
    using GridFormat::int8;
} // namespace Precision

using GridReader = GridFormat::Reader;

template<int order>
struct Order { static_assert(order > 0, "order must be > 0"); };

template<GridFormat::Concepts::Grid GridView, int order = 1>
class GridWriter
{
    using Grid = std::conditional_t<
        (order > 1),
        GridFormat::Dune::LagrangePolynomialGrid<GridView>,
        GridView
    >;
    using Cell = GridFormat::Cell<Grid>;
    using Vertex = typename GridView::template Codim<GridView::dimension>::Entity;
    using Element = typename GridView::template Codim<0>::Entity;
    using Coordinate = typename Element::Geometry::GlobalCoordinate;

    static_assert(
        order == 1 || __have_dune_local_functions,
        "dune-localfunctions required for higher-order output"
    );

 public:
    //! Constructor for static grid file formats
    template<typename Format>
    explicit GridWriter(const Format& fmt,
                        const GridView& gridView,
                        const Order<order>& = {})
    : grid_{makeGrid_(gridView)}
    , writer_{makeWriter_(fmt)}
    {}

    //! Constructor for time series
    template<typename Format>
    explicit GridWriter(const Format& fmt,
                        const GridView& gridView,
                        const std::string& filename,
                        const Order<order>& = {})
    : grid_{makeGrid_(gridView)}
    , writer_{makeWriter_(fmt, filename)}
    {}

    std::string write(const std::string& name) const
    { return writer_.write(name); }

    template<std::floating_point T>
    std::string write(T time) const
    { return writer_.write(time); }

    template<GridFormat::Concepts::CellFunction<GridView> F,
             GridFormat::Concepts::Scalar T = GridFormat::FieldScalar<std::invoke_result_t<F, Element>>>
    void addCellData(const std::string& name, F&& f, const GridFormat::Precision<T>& prec = {})
    {
        writer_.set_cell_field(name, [&, _f=std::move(f)] (const auto& cell) {
            return _f(GridFormat::Dune::Traits::Element<Grid>::get(grid_, cell));
        }, prec);
    }

    template<typename F>
        requires(std::is_lvalue_reference_v<F>)
    void addCellData(const std::string& name, F&& f)
    { GridFormat::Dune::set_cell_function(std::forward<F>(f), writer_, name); }

    template<typename F, GridFormat::Concepts::Scalar T>
        requires(std::is_lvalue_reference_v<F>)
    void addCellData(const std::string& name, F&& f, const GridFormat::Precision<T>& prec)
    { GridFormat::Dune::set_cell_function(std::forward<F>(f), writer_, name, prec); }

    template<GridFormat::Concepts::PointFunction<GridView> F,
             GridFormat::Concepts::Scalar T = GridFormat::FieldScalar<std::invoke_result_t<F, Vertex>>>
    void addPointData(const std::string& name, F&& f, const GridFormat::Precision<T>& prec = {})
    {
        static_assert(order == 1, "Point lambdas can only be used for order == 1");
        writer_.set_point_field(name, std::move(f), prec);
    }

    template<typename F>
        requires(std::is_lvalue_reference_v<F> )
    void addPointData(const std::string& name, F&& f)
    { GridFormat::Dune::set_point_function(std::forward<F>(f), writer_, name); }

    template<typename F, GridFormat::Concepts::Scalar T>
        requires(std::is_lvalue_reference_v<F> )
    void addPointData(const std::string& name, F&& f, const GridFormat::Precision<T>& prec)
    { GridFormat::Dune::set_point_function(std::forward<F>(f), writer_, name, prec); }

    void clear()
    { writer_.clear(); }

 private:
    Grid makeGrid_(const GridView& gv) const
    {
        if constexpr (order > 1)
            return Grid{gv, order};
        else
            return gv;
    }

    template<typename Format, typename... Args>
    auto makeWriter_(const Format& fmt, Args&&... args) const
    {
        const auto& comm = GridFormat::Dune::Traits::GridView<Grid>::get(grid_).comm();
        return makeParallelWriter_(fmt, comm, std::forward<Args>(args)...);
    }

    template<typename Format, typename... Args>
        requires(__have_mpi)
    auto makeParallelWriter_(const Format& fmt, __MPI_COMM comm, Args&&... args) const
    {
        return GridFormat::Parallel::size(comm) > 1
            ? GridFormat::Writer<Grid>{fmt, grid_, std::forward<Args>(args)..., comm}
            : GridFormat::Writer<Grid>{fmt, grid_, std::forward<Args>(args)...};
    }

    template<typename Format, typename... Args>
    auto makeParallelWriter_(const Format& fmt, No_Comm, Args&&... args) const
    { return GridFormat::Writer<Grid>{fmt, grid_, std::forward<Args>(args)...}; }

    Grid grid_;
    GridFormat::Writer<Grid> writer_;
};

} // namespace Dune::IO

#endif // Udune_io_HH
