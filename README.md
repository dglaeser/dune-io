## Playground for `Dune` wrapper around `GridFormat::Writer`

Clone this module with the flag `--recursive` in order to bring in the `GridFormat` library.

Contains a single executable (requires `dune-functions`):

```bash
make test_io
./test_io  # (optional) pass number of desired cells per direction as argument
```

TODO:

- For higher-order output, the higher-order grid is currently fully built upon construction of the writer, and an update mechanism (e.g. when the grid changes) is currently not exposed. One could think about building and destroying this grid when an actual write is performed, using the `update` and `clear` functions of the `GridFormat::Dune::LagrangePolynomialGrid`.
- Add `Dune::GridFactory` wrappers to the traits to facilitate reading in Dune grids from grid files with the GridFormat reader.
