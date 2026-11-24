# External theta convergence strategies

Theta loads these strategies when a configuration registers their shared libraries.

```bash
make -C pooled
```

`target_adaptive_mcmc_cov` starts chains from dispersed finite points, performs
discarded warmup passes on the real target, shrinks each empirical covariance
toward the Asimov covariance, and freezes the proposal before recording
production steps.

`pooled_mcmc_cov` is a second-stage strategy. First export a covariance from
independent training chains:

```bash
python3 pooled/export_theta_pooled_covariance.py \
  TRAINING_CHAIN_1.root TRAINING_CHAIN_2.root \
  --output pooled_covariance.txt
```

The text file records and checks the exact theta parameter order. New chains
then load the same covariance, draw independent dispersed starts around the
pooled training mean, and run a fixed symmetric Metropolis--Hastings proposal.
Never estimate the covariance from the production chains whose posterior will
be reported.

Neither plugin modifies or replaces the installed theta binary or libraries.

Add the relevant shared library to `options.plugin_files` alongside Theta's
core plugins. Use its registered name in the producer's `mcmc_strategy` block.
For a production run with a trained covariance, that block can be written as

```
mcmc_strategy = {
    type = "pooled_mcmc_cov";
    name = "production_chain";
    iterations = 50000;
    covariance-file = "pooled_covariance.txt";
};
```

The target-adaptive strategy uses the same block with type
`target_adaptive_mcmc_cov` and no `covariance-file` setting.
See [the Gaussian example](../examples/gaussoverflat-mcmc_chain_db.cfg) for a
producer that writes chains to a ROOT file.
