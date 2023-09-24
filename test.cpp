#include <filesystem>

#include <dune/grid/yaspgrid.hh>
#include <dune/grid/io/file/vtk/vtkwriter.hh>
#include <dune/functions/gridfunctions/analyticgridviewfunction.hh>

#include <dune/io/io.hh>
#include <dune/common/timer.hh>

template<typename Action>
auto measure(const Action& action) {
    Dune::Timer timer;
    action();
    timer.stop();
    return timer.elapsed();
}

template<typename T>
T copy(T t) {
    return t;
}

int main(int argc, char** argv) {
    const auto& helper = Dune::MPIHelper::instance(argc, argv);
    const int numCells = argc > 1 ? std::atoi(argv[1]) : 10;

    using Grid = Dune::YaspGrid<2>;
    using Element = typename Grid::template Codim<0>::Entity;

    Grid grid{{1.0, 1.0}, {numCells, numCells}};
    const auto& gridView = grid.leafGridView();
    const auto f = Dune::Functions::makeAnalyticGridViewFunction([] (const auto& x) { return x[0]; }, gridView);

    const auto format = Dune::IO::Format::vtu.with({
        .encoder = Dune::IO::Encoding::raw,
        .compressor = Dune::IO::Compression::none,
        .data_format = Dune::IO::VTK::DataFormat::appended,
        .coordinate_precision = Dune::IO::Precision::float32,
        .header_precision = Dune::IO::Precision::uint32
    });

    std::vector<double> cellData(gridView.size(0), 0.0);
    std::vector<double> pointData(gridView.size(2), 0.0);

    for (const auto& dir_entry : std::filesystem::directory_iterator{"."})
        if (dir_entry.path().string().ends_with("vtu"))
            std::filesystem::remove(dir_entry);

    const auto time_dune = measure([&] () {
        Dune::VTKWriter writer{gridView};
        writer.addCellData(cellData, "cfunc");
        writer.addCellData(cellData, "cdata");
        writer.addVertexData(pointData, "pfunc");
        writer.addVertexData(pointData, "pdata");
        writer.write("dune", Dune::VTK::appendedraw);
    });
    if (helper.rank() == 0)
        std::cout << "Dune write took " << time_dune << std::endl;

    const auto time_standard = measure([&] () {
        Dune::IO::GridWriter writer{format, gridView};
        writer.addCellData("cfunc", f, Dune::IO::Precision::float32);
        writer.addCellData("cdata", [&] (const auto& element) {
            return cellData.at(gridView.indexSet().index(element));
        }, Dune::IO::Precision::float32);
        writer.addPointData("pdata", [&] (const auto& vertex) {
            return pointData.at(gridView.indexSet().index(vertex));
        }, Dune::IO::Precision::float32);
        writer.addPointData("pfunc", f, Dune::IO::Precision::float32);
        writer.write("standard");
    });
    if (helper.rank() == 0)
        std::cout << "Standard write took " << time_standard << std::endl;

    const auto time_higher_order = measure([&] () {
        Dune::IO::GridWriter higherOrderWriter{format, gridView, Dune::IO::Order<2>{}};
        higherOrderWriter.addCellData("cfunc", f, Dune::IO::Precision::float32);
        higherOrderWriter.addCellData("cdata", [&] (const Element& element) {
            return cellData.at(gridView.indexSet().index(element));
        }, Dune::IO::Precision::float32);
        higherOrderWriter.addPointData("pfunc", f, Dune::IO::Precision::float32);
        higherOrderWriter.addPointData("pfunc_owning", copy(f), Dune::IO::Precision::float32);
        higherOrderWriter.write("higher_order");
    });
    if (helper.rank() == 0)
        std::cout << "Higher-order grid construction + write took " << time_higher_order << std::endl;

    return 0;
}
