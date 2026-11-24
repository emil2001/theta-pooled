# Theta with pooled MCMC

Theta is a framework for template-based statistical modeling and inference in high-energy physics by Thomas Müller, Jochen Ott, and Jeannine Wagner-Kuhr. This repository starts from the [KIT Subversion testing tag][svn] at revision 407 (March 2, 2015) and includes the pooled covariance and target-adaptive MCMC strategies.

## Build

The Make build uses a C++17 compiler, Boost, and SQLite. ROOT and `root-config` are needed for ROOT I/O. The covariance exporter also needs Python 3, NumPy, and PyROOT.

```bash
make -j4
make -C pooled
```

If your environment adds a system libconfig to `LD_LIBRARY_PATH`, give the bundled
library priority before running Theta.

```bash
export LD_LIBRARY_PATH="$PWD/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

Run `bin/theta` with a Theta configuration. See the [strategy documentation](pooled/README.md) for the pooled covariance workflow.

The build defaults to the system math library. The optional AMD LibM distribution is external. Theta's [upstream documentation](interface/doxygen-main.hpp) identifies its license as GPL. Bundled dependencies carry their own notices.

[svn]: https://ekptrac.physik.uni-karlsruhe.de/public/theta/tags/testing
